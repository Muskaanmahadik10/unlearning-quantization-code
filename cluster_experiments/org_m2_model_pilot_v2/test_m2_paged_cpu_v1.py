"""Mandatory cluster CPU tests; no skips, no claim of native CUDA certification."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import m2_paged_checkpoint_v1 as ck
from m2_pilot_evaluator_v1 import probability,oracles


class FakePaged(torch.optim.Optimizer):
    def __init__(self,params):
        super().__init__(params,{'lr':1e-5,'betas':(0.9,0.999),'eps':1e-8,'weight_decay':.01})
        self.is_paged=True
        self.args=SimpleNamespace(optim_bits=32,percentile_clipping=100,max_unorm=0,skip_zeros=False)
        self.mng=SimpleNamespace(uses_config_override=False)
    def init_state(self,group,p,gindex,pindex):
        self.state[p]={'step':0,'state1':torch.zeros_like(p,dtype=torch.float32),
                       'state2':torch.zeros_like(p,dtype=torch.float32)}


def fixture():
    model=torch.nn.Linear(4,5).bfloat16();opt=FakePaged(model.parameters())
    for i,p in enumerate(model.parameters()):
        opt.init_state(opt.param_groups[0],p,0,i)
        opt.state[p]['step']=2;opt.state[p]['state1'].fill_(.125);opt.state[p]['state2'].fill_(.25)
    return model,opt


class Tests(unittest.TestCase):
    def test_safetensors_roundtrip_and_rng(self):
        m,opt=fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(torch.cuda,'synchronize'):
            p=Path(d)/'cp';h=ck.save_checkpoint(torch,np,m,opt,p,{'test':'cpu-only'},2)
            expected=torch.rand(3).tolist();n=torch.nn.Linear(4,5).bfloat16();fresh=FakePaged(n.parameters())
            ck.load_checkpoint(torch,np,n,fresh,p,h,{'test':'cpu-only'},2)
            self.assertEqual(expected,torch.rand(3).tolist())
            self.assertEqual(ck.model_rows(torch,m),ck.model_rows(torch,n))
            self.assertEqual(ck.check_optimizer(torch,m,opt),ck.check_optimizer(torch,n,fresh))

    def test_checkpoint_identity_and_corruption(self):
        m,opt=fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(torch.cuda,'synchronize'):
            p=Path(d)/'cp';h=ck.save_checkpoint(torch,np,m,opt,p,{'test':'cpu-only'},2)
            n=torch.nn.Linear(4,5).bfloat16();fresh=FakePaged(n.parameters())
            with self.assertRaises(ValueError):ck.load_checkpoint(torch,np,n,fresh,p,h,{'test':'changed'},2)
            with self.assertRaises(ValueError):ck.load_checkpoint(torch,np,n,fresh,p,'0'*64,{'test':'cpu-only'},2)

    def test_moment_dtype_and_clean_boundary(self):
        m,opt=fixture();p=next(m.parameters())
        opt.state[p]['state1']=opt.state[p]['state1'].bfloat16()
        with self.assertRaises(ValueError):ck.check_optimizer(torch,m,opt)
        m,opt=fixture();next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):ck.save_checkpoint(torch,np,m,opt,Path(d)/'cp',{},2)

    def test_pinned_fp32_probability_primitive(self):
        torch.manual_seed(8);z=torch.randn(3,5,11);labels=torch.tensor([[-100,-100,2,3,4]]*3)
        class Fixed:
            device=torch.device('cpu')
            def __call__(self,**batch):return SimpleNamespace(logits=z[:len(batch['labels'])])
        for n in (2,1):
            ours=probability(torch,z[:n],labels[:n])
            reference=oracles(torch)['evaluate_probability'](Fixed(),{'labels':labels[:n]})
            for a,b in zip(ours,reference):
                for k in a:self.assertLessEqual(abs(a[k]-b[k]),1e-6)


if __name__=='__main__':unittest.main()
