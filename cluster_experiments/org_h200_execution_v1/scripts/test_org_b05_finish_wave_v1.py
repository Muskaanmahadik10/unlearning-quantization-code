import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import org_b05_finish_wave_v1 as wave

def sets():
    return {n:{'protected':[], 'alias_of':'full_packed' if n=='abstention' else None} for n in wave.cell.CONDITIONS}

def plan():
    return {'cells':[{'cell_id':c,'quantizer':q} for c,q in sorted(wave.EXPECTED)]}

def reuse():
    r={'ordinary_tofu_ga_seed42/rtn_int8/'+n:{} for n in wave.cell.CONDITIONS if n!='abstention'}
    r.update({'ordinary_tofu_ga_seed42/'+q+'/pair':{} for q in ('nf4','gptq_int4')})
    return r

class Tests(unittest.TestCase):
    def test_23_unique_tasks_and_budget(self):
        with patch.object(wave.cell,'context',return_value=(None,[],sets())):
            tasks,conditions=wave.make_tasks(plan(),reuse())
        self.assertEqual(len(tasks),23)
        self.assertEqual(sum(t['group']=='gptq' for t in tasks),9)
        self.assertEqual(sum(t['group']=='other' for t in tasks),14)
        self.assertEqual(sum(t['limit_seconds'] for t in tasks)+wave.FINALIZER_SECONDS,58500)
        self.assertEqual(len({t['folder'] for t in tasks}),23)
        self.assertEqual(len(conditions),6)
    def test_safe_alias_deduplicates(self):
        s=sets(); s['safe_looking_singleton']['alias_of']='full_packed'
        with patch.object(wave.cell,'context',return_value=(None,[],s)):
            tasks,_=wave.make_tasks(plan(),reuse())
        self.assertFalse(any(t['condition']=='safe_looking_singleton' for t in tasks))
    def test_missing_cell_rejected(self):
        p=plan(); p['cells'].pop()
        with self.assertRaises(RuntimeError): wave.make_tasks(p,reuse())
    def test_accounting(self):
        t='123_0|COMPLETED|10|cpu=8,gres/gpu=1\n123_1|FAILED|20|gres/gpu=1'
        r=wave.accounting(t,{'123_0','123_1'})
        self.assertEqual(sum(x['seconds'] for x in r.values()),30)
        for bad in (t.splitlines()[0],t+'\n'+t,t.replace('FAILED','RUNNING'),t.replace('gpu=1','gpu=8')):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError): wave.accounting(bad,{'123_0','123_1'})
    def test_failed_child_has_no_completion_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'condition'
            task={'group':'gptq','cell':'ordinary_tofu_ga_seed42','quantizer':'gptq_int4','condition':'pair','folder':str(folder)}
            with patch.object(wave,'load_wave',return_value={'tasks':[task]}),patch.object(wave.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'worker')):
                with self.assertRaises(subprocess.CalledProcessError): wave.run('gptq',0)
            self.assertFalse((folder/'condition_complete.json').exists())
    def test_duplicate_writer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            task={'group':'other','folder':d}
            with patch.object(wave,'load_wave',return_value={'tasks':[task]}),patch.object(wave.subprocess,'run') as child:
                with self.assertRaises(FileExistsError): wave.run('other',0)
                child.assert_not_called()
    def test_failed_array_writes_incomplete_no_scientific_pass(self):
        with tempfile.TemporaryDirectory() as d,patch.object(wave,'ROOT',Path(d)):
            wave.life.atomic(Path(d)/'submissions.json',{'arrays':{'gptq':'123'}})
            task={'group':'gptq','folder':str(Path(d)/'missing')}
            with patch.object(wave,'load_wave',return_value={'tasks':[task]}),patch.object(wave,'query_jobs',return_value='123_0|FAILED|5|gres/gpu=1'):
                with self.assertRaises(RuntimeError): wave.aggregate()
            r=wave.life.read(Path(d)/'final.json')
            self.assertEqual(r['status'],'INCOMPLETE')
            self.assertFalse(r['atlas_production_authorized'])
    def test_artifact_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d); (folder/'artifact').mkdir(); f=folder/'artifact/model'; f.write_bytes(b'good')
            wave.life.atomic(folder/'artifact_hashes.json',{'model':wave.life.sha256_file(f)})
            rows=[{'index':i,'sample_sha256':str(i),'encoding_sha256':str(i),'tokens':1,'loss_sum':1.0} for i in range(128)]
            for p in wave.PHASES:
                dest=folder/'emulated' if p=='emulated' else folder
                wave.life.atomic(dest/(p+'_behavior.json'),{r:{'rows':rows} for r in ('forget','retain')})
            f.write_bytes(b'bad')
            with patch.object(wave.prior,'validate'),self.assertRaises(RuntimeError): wave.validate(folder,True)
    def test_submission_timeout_retains_exclusive_attempt(self):
        import submit_org_b05_finish_wave_v1 as submit
        with tempfile.TemporaryDirectory() as d,patch.object(submit,'ROOT',Path(d)),patch.object(submit.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',45)) as call:
            with self.assertRaises(subprocess.TimeoutExpired): submit.submit('other',[])
            with self.assertRaises(FileExistsError): submit.submit('other',[])
            self.assertEqual(call.call_count,1)
    def test_full_aggregation_and_negative_scientific_gate(self):
        for negative in (False,True):
            with self.subTest(negative=negative),tempfile.TemporaryDirectory() as d:
                root=Path(d); planfile=root/'plan.json'; wave.life.atomic(planfile,{'status':'fixture'})
                w={'tasks':[],'conditions':{},'reused':{},'checkpoint_files':{}}
                for cid,q in sorted(wave.EXPECTED):
                    key=cid+'/'+q; w['conditions'][key]=sets()
                    for n in wave.cell.CONDITIONS:
                        if n=='abstention': continue
                        folder=root/'data'/(cid+'__'+q)/n
                        value={'bf16':1.5,'full_packed':2.,'safe_looking_singleton':3.,'risky_looking_singleton':4.,'pair':5.}[n]
                        for phase,scale in (('reload_b',value),('emulated',value)):
                            if negative and cid=='adverse_muse_gdr_seed2024' and q=='gptq_int4' and n=='pair' and phase=='reload_b': scale=1.
                            rows=[{'index':i,'sample_sha256':str(i),'encoding_sha256':str(i),'tokens':1,'loss_sum':scale} for i in range(128)]
                            dest=folder/'emulated' if phase=='emulated' else folder
                            wave.life.atomic(dest/(phase+'_behavior.json'),{r:{'rows':rows} for r in ('forget','retain')})
                        w['reused'][key+'/'+n]={'folder':str(folder),'json_hashes':wave.inventory(folder)}
                wave.life.atomic(root/'submissions.json',{'arrays':{}})
                wave.life.atomic(root/'wave.json',w)
                with patch.object(wave,'ROOT',root),patch.object(wave,'PLAN',planfile),patch.object(wave,'load_wave',return_value=w),patch.object(wave,'query_jobs',return_value=''),patch.object(wave,'validate'):
                    wave.aggregate()
                r=wave.life.read(root/'final.json')
                self.assertEqual(r['status'],'EXECUTION_COMPLETE')
                self.assertEqual(r['conditions'],36)
                self.assertEqual(len((root/'condition_results.tsv').read_text().splitlines()),37)
                self.assertEqual(r['atlas_lock_preparation_allowed'],not negative)
                self.assertFalse(r['atlas_production_authorized'])

if __name__=='__main__': unittest.main()
