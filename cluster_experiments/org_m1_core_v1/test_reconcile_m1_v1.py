import copy
import unittest
from reconcile_m1_smokes_v1 import accounting, phase_check, behavior_check, ARMS

class Tests(unittest.TestCase):
    def test_accounting_all_attempts(self):
        rows=[f'357253_{i}|CANCELLED|0:0|294|gres/gpu=1' for i in (3,4,5)]
        rows += [f'357308_{i}|COMPLETED|0:0|400|gres/gpu=1' for i in (3,4,5)]
        rows += ['357360_0|FAILED|1:0|99|gres/gpu=1','357360_1|COMPLETED|0:0|440|gres/gpu=1','357360_2|COMPLETED|0:0|1021|gres/gpu=1','357529|COMPLETED|0:0|407|gres/gpu=1']
        self.assertEqual(len(accounting('\n'.join(rows))),10)
        for bad in (rows[:-1],rows+[rows[0]],[r.replace('COMPLETED','RUNNING') for r in rows],[r.replace('gpu=1','gpu=8') for r in rows]):
            with self.assertRaises(ValueError): accounting('\n'.join(bad))
    def test_behavior_identity_and_strict_threshold(self):
        a={role:{'rows':[{'index':i,'tokens':2,'sample_sha256':str(i),'encoding_sha256':'x','loss_sum':1.} for i in range(128)]} for role in ('forget','retain')}
        self.assertEqual(behavior_check(a,copy.deepcopy(a)),0.)
        for field,value in [('encoding_sha256','other'),('loss_sum',2.),('tokens',0)]:
            b=copy.deepcopy(a); b['forget']['rows'][0][field]=value
            with self.assertRaises(ValueError): behavior_check(a,b)
    def test_exact_arm_coverage(self):
        cell={'cell':'x','a':'m.a.weight','b':'m.b.weight'}
        r={'status':'PASS','cell':'x','phase':'produce','source_lock_sha256':'sha','requantized':False,'core_launch_authorized':False,'seconds':1.,'load_seconds':.1,'arms':{}}
        for arm in ARMS:
            r['arms'][arm]={'status':'PASS','targets':[] if arm=='N0' else ['m.'+s for s in ('a','b') if s in arm.split('_')[1]],'dose':None if arm=='N0' else {'Z':0.,'M':.5,'R':1.}[arm[0]],'seconds':1.,'behavior':{} if arm in ('N0','Z_ab') else None}
        phase_check(r,cell,'sha','produce')
        for kind in ('missing','dose','source'):
            b=copy.deepcopy(r)
            if kind=='missing': b['arms'].pop('R_a')
            elif kind=='dose': b['arms']['R_a']['dose']=.99
            else: b['source_lock_sha256']='bad'
            with self.assertRaises(ValueError): phase_check(b,cell,'sha','produce')

if __name__=='__main__': unittest.main()
