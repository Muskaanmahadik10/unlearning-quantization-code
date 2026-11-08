import copy
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
import m2_paged_checkpoint_v1 as ck
from m2_u_common_v1 import canonical
from m2_u_model_io_v1 import install_model,separate_teacher
from test_m2_training_torch_v1 import Tiny,batch
from m2_training_core_v1 import update

class Tests(unittest.TestCase):
    def fixture(self,p,model):
        save_file({k:v.detach().cpu().contiguous() for k,v in ck.named_tensors(model).items()},str(p/'model.safetensors'))
        return {'model_tensors':ck.model_rows(torch,model),'aliases':ck.aliases(model)}
    def test_model_only_install_exact_and_optimizer_fresh(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=Tiny();b=Tiny();m=self.fixture(p,a)
            install_model(torch,b,p,m,canonical(m['model_tensors']))
            self.assertEqual(ck.model_rows(torch,a),ck.model_rows(torch,b))
            opt=torch.optim.AdamW(b.parameters());self.assertEqual(len(opt.state),0)
            self.assertFalse((p/'optimizer.safetensors').exists())
    def test_bad_digest_and_dtype_fail(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=Tiny();m=self.fixture(p,a)
            with self.assertRaises(ValueError):install_model(torch,Tiny(),p,m,'wrong')
            with self.assertRaises(ValueError):install_model(torch,Tiny().double(),p,m,canonical(m['model_tensors']))
    def test_shared_teacher_rejected(self):
        a=Tiny()
        with self.assertRaises(ValueError):separate_teacher(a,a)
        b=copy.deepcopy(a);b.embed.weight=a.embed.weight
        with self.assertRaises(ValueError):separate_teacher(a,b)
    def test_loaded_teacher_stays_exact_while_student_updates(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=Tiny().double();b=Tiny().double();m=self.fixture(p,a)
            install_model(torch,b,p,m,canonical(m['model_tensors']));separate_teacher(a,b)
            before=ck.model_rows(torch,b);opt=torch.optim.AdamW(a.parameters(),lr=.01)
            update(torch,a,opt,[{'anchor':batch(),'retain':batch()}],reference=b)
            self.assertEqual(before,ck.model_rows(torch,b));self.assertNotEqual(before,ck.model_rows(torch,a))
            self.assertTrue(all(not x.requires_grad and x.grad is None for x in b.parameters()))
if __name__=='__main__':unittest.main()
