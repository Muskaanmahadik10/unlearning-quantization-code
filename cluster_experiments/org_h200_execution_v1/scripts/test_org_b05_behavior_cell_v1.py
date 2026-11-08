import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_org_b05_behavior_cell_v1 as runner
from org_b05_behavior_wave_contract_v1 import CONDITIONS


class OrchestrationTests(unittest.TestCase):
    def test_all_conditions_and_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=root/'plan'; auth=root/'auth'; sources=root/'sources'
            for p in (plan,auth,sources): p.write_text('{}')
            output=root/'output'
            sets={c:{'protected':[],'alias_of':'full_packed' if c=='abstention' else None} for c in CONDITIONS}
            calls=[]
            def fake(command,check):
                self.assertTrue(check)
                phase=command[command.index('--phase')+1]
                dest=Path(command[command.index('--output')+1])
                calls.append((command[command.index('--condition')+1],phase))
                behavior={r:{'rows':[{'index':i,'sample_sha256':str(i),'encoding_sha256':str(i),
                            'tokens':1,'loss_sum':2.} for i in range(128)]} for r in ('forget','retain')}
                (dest/(phase+'_behavior.json')).write_text(json.dumps(behavior))
                (dest/(phase+'.json')).write_text(json.dumps({'pid':len(calls),'status':'PASS'}))
            argv=['runner','--plan',str(plan),'--authorization',str(auth),'--source-manifest',str(sources),
                  '--output',str(output),'--cell','ordinary','--quantizer','rtn_int8']
            with patch('sys.argv',argv), patch.object(runner,'authorize'), patch.object(runner,'context',return_value=({},[],sets)), patch.object(runner.subprocess,'run',side_effect=fake):
                runner.main()
            result=json.loads((output/'final.json').read_text())
            self.assertEqual(len(calls),20)
            self.assertEqual(set(result['conditions']),set(CONDITIONS))
            self.assertEqual(result['conditions']['abstention']['record'],result['conditions']['full_packed'])
            self.assertEqual(result['transport']['status'],'INCONCLUSIVE')
            self.assertFalse(result['production_authorized'])

    def test_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'output'
            sets={c:{'protected':[],'alias_of':None} for c in CONDITIONS}
            argv=['runner','--plan','p','--authorization','a','--source-manifest','s',
                  '--output',str(output),'--cell','ordinary','--quantizer','rtn_int8']
            with patch('sys.argv',argv), patch.object(runner,'authorize'), patch.object(runner,'context',return_value=({},[],sets)), patch.object(runner.subprocess,'run',side_effect=RuntimeError('child failed')):
                with self.assertRaisesRegex(RuntimeError,'child failed'): runner.main()
            self.assertFalse((output/'final.json').exists())


if __name__=='__main__': unittest.main()
