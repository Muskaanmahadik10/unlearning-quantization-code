"""Must run with Torch installed; absent Torch is an error, never a skipped PASS."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
import m2_training_core_v1 as c

torch.set_num_threads(1)


class Tiny(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.embed=torch.nn.Embedding(11,5)
        self.dropout=torch.nn.Dropout(dropout)
        self.head=torch.nn.Linear(5,11)

    def forward(self,input_ids,labels,attention_mask=None):
        z=self.head(self.dropout(self.embed(input_ids)))
        loss=F.cross_entropy(z[:,:-1,:].transpose(1,2), labels[:,1:],ignore_index=-100)
        return SimpleNamespace(logits=z,loss=loss)


def batch():
    return {'input_ids':torch.tensor([[1,2,3,4,5],[1,7,5,3,0]]),
            'labels':torch.tensor([[-100,-100,3,4,5],[-100,-100,-100,3,0]]),
            'attention_mask':torch.ones(2,5,dtype=torch.bool)}


def official_utils():
    p=Path(__file__).parent/'upstream_trainer_utils.py'
    spec=importlib.util.spec_from_file_location('m2_pinned_loss_oracle',p)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Tests(unittest.TestCase):
    def setUp(self): torch.manual_seed(408)

    def test_npo_loss_and_gradient_match_pinned_upstream(self):
        m=Tiny().double(); reference=c.freeze_reference(copy.deepcopy(m)); x=batch()
        with torch.no_grad(): m.head.weight.add_(0.02*torch.randn_like(m.head.weight))
        ours=c.npo_loss(torch,m(**x).logits,reference(**x).logits,x['labels'])
        upstream,_=official_utils().compute_dpo_loss(m,reference,lose_inputs=x,beta=.1)
        torch.testing.assert_close(ours,upstream,rtol=0,atol=1e-12)
        ga=torch.autograd.grad(ours,tuple(m.parameters()))
        gb=torch.autograd.grad(upstream,tuple(m.parameters()))
        for a,b in zip(ga,gb): torch.testing.assert_close(a,b,rtol=0,atol=1e-12)

    def test_npo_initial_loss_and_direction(self):
        z=torch.randn(2,5,11,dtype=torch.float64,requires_grad=True); y=batch()['labels']
        loss=c.npo_loss(torch,z,z.detach(),y)
        self.assertAlmostEqual(loss.item(),20*math.log(2),places=12)
        grad=torch.autograd.grad(loss,z)[0]
        before=c.sequence_nll(torch,z,y).mean()
        after=c.sequence_nll(torch,z-.001*grad,y).mean()
        self.assertGreater(after.item(),before.item())

    def test_finite_difference_gradient(self):
        z=torch.randn(1,3,5,dtype=torch.float64,requires_grad=True)
        teacher=torch.randn_like(z); labels=torch.tensor([[-100,2,3]])
        self.assertTrue(torch.autograd.gradcheck(lambda x:c.npo_loss(torch,x,teacher,labels),
                                                (z,),eps=1e-6,atol=1e-5))

    def test_teacher_frozen_and_unchanged(self):
        m=Tiny().double(); teacher=c.freeze_reference(copy.deepcopy(m))
        identity=c.state_sha(torch,teacher)
        optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
        c.update(torch,m,optimizer,[{'anchor':batch(),'retain':batch()}],reference=teacher)
        self.assertEqual(identity,c.state_sha(torch,teacher))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_npo_plus_retain_matches_oracle(self):
        m=Tiny().double(); teacher=c.freeze_reference(copy.deepcopy(m)); x=batch()
        expected,_=official_utils().compute_dpo_loss(m,teacher,lose_inputs=x,beta=.1)
        expected=expected+m(**x).loss
        actual=c.objective(torch,m,{'anchor':x,'retain':x},reference=teacher)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)

    def test_full_and_partial_accumulation_gradients(self):
        for count in (4,2):
            a=Tiny().double(); b=copy.deepcopy(a)
            oa=torch.optim.SGD(a.parameters(),lr=.01); ob=torch.optim.SGD(b.parameters(),lr=.01)
            micros=[{'anchor':batch()} for _ in range(count)]
            for i,micro in enumerate(micros):
                # Unequal label counts ensure this checks mean-of-microbatch loss,
                # not silently changed global-token normalization.
                micro['anchor']['labels'][0,1:min(2+i,5)]=-100
            c.update(torch,a,oa,micros,max_grad_norm=1e6)
            sum(b(**m['anchor']).loss for m in micros).div(count).backward(); ob.step()
            for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa,pb,rtol=0,atol=1e-12)

    def test_resume_exact_with_dropout_and_optimizer_state(self):
        a=Tiny(dropout=.2).double(); oa=torch.optim.AdamW(a.parameters(),lr=.01)
        for _ in range(2): c.update(torch,a,oa,[{'anchor':batch()}])
        oa.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'own_resume.pt'
            digest=c.save_resume(torch,p,a,oa,2,'contract','schedule')
            for _ in range(2): c.update(torch,a,oa,[{'anchor':batch()}])
            b=Tiny(dropout=.2).double(); ob=torch.optim.AdamW(b.parameters(),lr=.01)
            self.assertEqual(c.load_resume(torch,p,digest,b,ob,'contract','schedule'),2)
            for _ in range(2): c.update(torch,b,ob,[{'anchor':batch()}])
            self.assertEqual(c.state_sha(torch,a),c.state_sha(torch,b))
            with self.assertRaises(ValueError): c.load_resume(torch,p,'wrong',b,ob,'contract','schedule')
            with self.assertRaises(ValueError): c.load_resume(torch,p,digest,b,ob,'wrong','schedule')

    def test_padding_keeps_official_eos_semantics(self):
        examples=[{'input_ids':[1,2,3],'labels':[-100,2,3]}, {'input_ids':[1,3],'labels':[-100,3]}]
        out=c.collate(torch,examples,pad_id=3)
        self.assertEqual(out['input_ids'].tolist(),[[1,2,3],[1,3,3]])
        self.assertEqual(out['attention_mask'].tolist(),[[True,True,False],[True,False,False]])
        self.assertEqual(out['labels'].tolist(),[[-100,2,3],[-100,3,-100]])

    def test_nonfinite_and_unlabeled_rejected(self):
        z=torch.randn(2,5,11); y=batch()['labels']
        z[0,0,0]=float('nan')
        with self.assertRaises(ValueError): c.sequence_nll(torch,z,y)
        with self.assertRaises(ValueError): c.sequence_nll(torch,torch.zeros(2,5,11),torch.full_like(y,-100))


if __name__=='__main__': unittest.main()
