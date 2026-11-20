"""CPU checks without PVT weights, timm, torchvision or dataset access."""
import importlib.util
import unittest
from pathlib import Path
import torch

path=Path(__file__).resolve().parents[1]/'model/scale_prior.py'
spec=importlib.util.spec_from_file_location('scale_prior',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ScalePrior=module.ScalePrior


class ScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x=torch.randn(2,32,4,4)
        self.ids=torch.stack([torch.zeros(4,4,dtype=torch.long),
            torch.arange(4).view(4,1).expand(4,4),torch.arange(16).reshape(4,4)]).repeat(2,1,1,1)

    def test_fixed20_matches_original_membership(self):
        prior=ScalePrior(mode='fixed20'); result=prior(self.x,self.ids)
        r=self.ids[:,1].flatten(1)
        self.assertTrue(torch.equal(result,r[:,:,None].eq(r[:,None,:]).float()))

    def test_uniform_and_initial_adaptive_match(self):
        a=ScalePrior(mode='adaptive'); u=ScalePrior(mode='uniform')
        self.assertTrue(torch.allclose(a(self.x,self.ids),u(self.x,self.ids)))
        self.assertTrue(torch.allclose(a.last_weights.sum(1),torch.ones(2)))

    def test_duplicate_partitions_and_label_permutation(self):
        ids=self.ids.clone(); ids[:,2]=ids[:,1]*7+19
        a=ScalePrior(); a(self.x,ids)
        self.assertTrue(torch.equal(a.last_weights[:,2],torch.zeros(2)))
        self.assertTrue(torch.allclose(a.last_weights[:,:2],torch.full((2,2),0.5)))
        ids[:]=4; a(self.x,ids)
        self.assertTrue(torch.equal(a.last_weights[:,1],torch.ones(2)))

    def test_deduplicate_after_resize(self):
        ids=torch.zeros(2,3,8,8,dtype=torch.long); ids[:,2,1::2,1::2]=1
        a=ScalePrior(); a(self.x,ids)
        self.assertEqual(a.last_active.sum().item(),2)

    def test_gate_gradients_and_image_dependence(self):
        a=ScalePrior(); opt=torch.optim.SGD(a.parameters(),lr=0.1)
        target=torch.randn(2,16,16)
        for _ in range(3):
            opt.zero_grad(); loss=(a(self.x,self.ids)*target).sum(); loss.backward()
            self.assertTrue(torch.isfinite(a.gate[-1].weight.grad).all())
            self.assertGreater(a.gate[-1].weight.grad.abs().sum().item(),0)
            opt.step()
        a(self.x,self.ids)
        self.assertFalse(torch.allclose(a.last_weights[0],a.last_weights[1]))

    def test_invalid_inputs(self):
        a=ScalePrior()
        with self.assertRaises(ValueError): a(self.x,self.ids.float())
        with self.assertRaises(ValueError): a(self.x,self.ids[:,0])


if __name__=='__main__': unittest.main()
