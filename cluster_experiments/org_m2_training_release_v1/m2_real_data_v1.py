"""Complete record/token routing; CPU binding never evaluates real outcomes."""
import hashlib
import json
from pathlib import Path
from m2_stage_common_v1 import require,canonical,save,read,sha,validate_binding,ROOT,PILOT

ROUTES={'forget':'forget10_perturbed','retain':'retain_perturbed','ra':'real_authors',
        'wf':'world_facts','holdout':'holdout10','retain_all':'retain90'}

def qa(row):return hashlib.sha256(json.dumps([row['question'],row['answer']],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def raw_data(binding,root):
    return {Path(r['path']).stem:[json.loads(x) for x in (root/r['path']).read_text(encoding='utf-8').splitlines() if x.strip()] for r in binding['data_files']}

def route(raw):
    result={}
    for panel,name in ROUTES.items():
        rows=raw[name]
        if panel in ('ra','wf'):
            pert=raw['real_authors_perturbed' if panel=='ra' else 'world_facts_perturbed']
            require(len(pert)==len(rows) and [qa(r) for r in rows]==[qa(r) for r in pert],'correct/perturbed row alignment '+panel)
            rows=[{**r,'perturbed_answer':p['perturbed_answer']} for r,p in zip(rows,pert)]
        output=[]
        for i,r in enumerate(rows):
            answers={'answer':[r['answer']]}
            if panel in ('forget','retain'):answers['paraphrase']=[r['paraphrased_answer']]
            if panel in ('forget','retain','ra','wf'):
                require(isinstance(r['perturbed_answer'],list) and len(r['perturbed_answer'])>=1,'perturbed alternatives')
                answers['wrong']=r['perturbed_answer']
            output.append({'index':str(i),'qa_sha256':qa(r),'question':r['question'],'answers':answers})
        require(len({r['qa_sha256'] for r in output})==len(output),'duplicate exact QA '+panel)
        result[panel]=output
    return result

def prepare(torch,values,output):
    from transformers import AutoTokenizer
    from m2_training_core_v1 import encode_qa
    from m2_training_cpu_preflight_v1 import load_module
    plan,base,binding,tokens,schedules,cal,design=values
    raw=raw_data(binding,ROOT/'org_m2_evaluator_setup_v1');routed=route(raw)
    tokenizer=AutoTokenizer.from_pretrained(base['snapshot_path'],local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    require(tokenizer.pad_token_id==128009,'tokenizer pad identity')
    oracle=load_module('m2_real_data_oracle',PILOT/'upstream_data_utils.py')
    template={'apply_chat_template':True,'system_prompt':design['prompt']['system'],'date_string':design['prompt']['date']}
    variants=0
    for panel,rows in routed.items():
        for row in rows:
            row['tokens']={}
            for kind,answers in row.pop('answers').items():
                entries=[]
                for answer in answers:
                    e=encode_qa(tokenizer,row['question'],answer,512)
                    expected=oracle.preprocess_chat_instance(tokenizer,template,[row['question']],[answer],max_length=512)
                    require(all(e[k]==expected[k].tolist() for k in ('input_ids','labels','attention_mask')),'real tokenizer parity')
                    gen=oracle.preprocess_chat_instance(tokenizer,template,[row['question']],[answer],max_length=512,predict_with_generate=True)
                    require(e['prompt_ids']==gen['input_ids'].tolist(),'generation prompt parity')
                    entries.append(e);variants+=1
                row['tokens'][kind]=entries
        print('M2_REAL_DATA_BOUND='+panel+':'+str(len(rows)),flush=True)
    identity=lambda panel:{r['qa_sha256'] for r in routed[panel]}
    f={qa(r) for r in raw['forget10']};r={qa(x) for x in raw['retain90']};full={qa(x) for x in raw['full']}
    summary={'counts':{k:len(v) for k,v in routed.items()},'all_variant_token_parity':True,
       'official_forget_exact_qa_matches_training_forget':identity('forget')==f,
       'retain_eval_subset_of_training_retain':identity('retain')<=r,
       'holdout_training_overlap':len(identity('holdout')&full),'evaluated_model_outcomes':0,
       'encoded_variants':variants,'dataset_manifest_sha256':canonical(binding['data_files']),
       'panel_record_identity_sha256':canonical({k:[r['qa_sha256'] for r in v] for k,v in routed.items()}),
       'pad_id':tokenizer.pad_token_id,'generation':design['evaluation']['generation']}
    validate_binding(summary)
    save(output/'EVALUATION_TOKENS.json',routed)
    summary['tokens_sha256']=sha(output/'EVALUATION_TOKENS.json');save(output/'DATA_BINDING.json',summary)
    return summary
