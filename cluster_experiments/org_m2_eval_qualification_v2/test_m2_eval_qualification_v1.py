import copy
import json
from pathlib import Path
import tempfile
import unittest
from m2_eval_qualification_common_v1 import *

def toy_cal():
    records=[]
    for j in range(128):
        ids=[1,2,3+j%3,4,0]
        records.append({'qa_sha256':str(j),'candidate_label':'fixture','input_ids':ids,'labels':[-100,-100]+ids[2:],'attention_mask':[1]*5,'prompt_ids':[1,2]})
    return {'records':records}

class Tests(unittest.TestCase):
    def test_exact_four_roles(self):self.assertEqual(ROLES,('B','O','R','U'))
    def test_fixture_scope(self):
        p=fixture(toy_cal());self.assertEqual(set(p),set(PANELS));self.assertEqual(len(p['forget']),3)
    def test_fixture_only_calibration_ids(self):
        p=fixture(toy_cal())
        self.assertEqual({r['qa_sha256'] for rows in p.values() for r in rows},{'0','1','2'})
    def test_options_all_covered(self):
        p=fixture(toy_cal())
        for k in ('forget','retain','ra','wf'):self.assertTrue(all(len(r['tokens']['wrong'])==6 for r in p[k]))
    def test_nonmember_is_explicit_fixture(self):self.assertEqual(fixture(toy_cal())['forget'][0]['tokens']['answer'],fixture(toy_cal())['holdout'][0]['tokens']['answer'])
    def test_calibration_missing_rejected(self):
        c=toy_cal();c['records'].pop()
        with self.assertRaises(ValueError):fixture(c)
    def test_calibration_duplicate_rejected(self):
        c=toy_cal();c['records'][1]=c['records'][0]
        with self.assertRaises(ValueError):fixture(c)
    def test_fixture_no_source_mutation(self):
        c=toy_cal();before=copy.deepcopy(c);p=fixture(c);p['forget'][0]['tokens']['answer'][0]['input_ids'][0]=999;self.assertEqual(c,before)
    def test_score_tolerance(self):self.assertLessEqual(compare({'x':1.0},{'x':1.0+5e-7}),1e-6)
    def test_score_large_delta_rejected(self):
        with self.assertRaises(ValueError):compare(1.0,1.00001)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):compare(float('nan'),float('nan'))
    def test_token_ids_exact(self):
        with self.assertRaises(ValueError):compare([1,2],[1,3])
    def test_missing_row_rejected(self):
        with self.assertRaises(ValueError):compare({'0':1.},{'1':1.})
    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):read(p)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';save(p,{})
            with self.assertRaises(FileExistsError):save(p,{})
    def test_path_guard(self):
        with tempfile.TemporaryDirectory() as d:
            for p in ('../x','/x','x\\a','C:x','x/../a'):
                with self.assertRaises(ValueError):confined(Path(d),p)
    def test_manifest_hash_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('a');row={'path':'x','bytes':1,'sha256':sha(p)};verify(Path(d),[row]);p.write_text('b')
            with self.assertRaises(ValueError):verify(Path(d),[row])
    def test_job_parsing(self):
        self.assertEqual(job_id('123;cluster\n'),'123')
        for v in ('','Submitted 123','123\n124','0'):
            with self.assertRaises(ValueError):job_id(v)
    def test_accounting_complete(self):
        txt='\n'.join('123_'+str(i)+'|COMPLETED|0:0|300|cpu=8,gres/gpu=1' for i in range(4));self.assertEqual(sum(accounting(txt,'123').values()),1200)
    def test_accounting_fail_closed(self):
        good='\n'.join('123_'+str(i)+'|COMPLETED|0:0|300|cpu=8,gres/gpu=1' for i in range(4))
        for bad in (good.replace('COMPLETED','RUNNING'),good.replace('300','901'),good.replace('gpu=1','gpu=2'),good.split('\n')[0],good+'\n'+good.split('\n')[0]):
            with self.assertRaises(ValueError):accounting(bad,'123')
    def test_slurm_resource_contract(self):
        s=(Path(__file__).parent/'m2_eval_qualification.slurm').read_text()
        for v in ('--array=0','--gres=gpu:1','--mem=200G','--time=00:15:00','module load cuda/12.6','-I -B'):self.assertIn(v,s)
    def test_cpu_before_submit(self):
        s=(Path(__file__).parent/'m2_eval_qualification_submit_v1.py').read_text()
        self.assertLess(s.index('result.wasSuccessful()'),s.index("['sbatch','--parsable'"));self.assertIn('SUBMISSION_UNKNOWN.json',s)
    def test_no_training_or_download(self):
        s=(Path(__file__).parent/'m2_eval_qualification_v1.py').read_text()
        for bad in ('.backward(','.step(','.save_pretrained(','.quantize(', 'snapshot_download('):self.assertNotIn(bad,s)
    def test_eight_record_acceleration_fixed(self):
        s=(Path(__file__).parent/'m2_eval_qualification_v1.py').read_text();self.assertIn("cal['records'][:8]",s)
        self.assertIn('automatic_batch_selection', (Path(__file__).parent/'m2_eval_qualification_oracle_v1.py').read_text())

if __name__=='__main__':unittest.main()
