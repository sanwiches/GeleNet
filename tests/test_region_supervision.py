import sys,unittest,tempfile,json,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.io import savemat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from region_targets import propagate,quartile_levels
from region_supervision_loss import region_supervision_loss
from weak_loss import partial_bce_with_logits


class RegionTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(quartile_levels(60),[15,30,45])
        self.assertEqual(quartile_levels(10),[2,5,7])
        self.assertEqual(quartile_levels(1),[1,1,1])
        with self.assertRaises(ValueError): quartile_levels(0)

    def test_native_propagation(self):
        r=np.array([[0,0,1,1],[2,2,3,3]])
        y=np.array([[1,0,0,0],[1,0,0,0]],dtype=np.uint8)
        m=np.array([[1,0,1,0],[1,1,0,0]],dtype=np.uint8)
        labels,mask,stats=propagate(r,y,m)
        self.assertEqual(mask.tolist(),[[0,1,0,1],[0,0,0,0]])
        self.assertEqual(labels[0,1],1); self.assertEqual(labels[0,3],0)
        self.assertEqual(stats['conflict'],1); self.assertEqual(stats['empty'],1)
        self.assertFalse((mask*m).any())

    def test_no_annotations_do_not_become_background(self):
        y,m,_=propagate(np.zeros((2,2),dtype=int),np.zeros((2,2)),np.zeros((2,2)))
        self.assertEqual(m.sum(),0)

    def test_weighted_terms_and_gradient(self):
        z=torch.tensor([[[[0.2,-0.6]]]],requires_grad=True)
        target=torch.tensor([[[[1.,0.]]]]); valid=torch.tensor([[[[1.,0.]]]])
        ys=torch.tensor([[[[0.,1.]],[[1.,0.]],[[1.,1.]]]])
        ms=torch.tensor([[[[0.,1.]],[[0.,1.]],[[0.,1.]]]])
        loss,terms=region_supervision_loss(z,target,valid,ys,ms)
        expected=sum(w*term for w,term in zip([.5,.1,.15,.25],terms))
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward(); self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(),0)

    def test_empty_terms_are_zero_without_renormalization(self):
        z=torch.zeros(1,1,2,2,requires_grad=True); y=torch.ones_like(z)
        loss,terms=region_supervision_loss(z,y,y,torch.zeros(1,3,2,2),torch.zeros(1,3,2,2))
        self.assertTrue(torch.allclose(loss,.5*partial_bce_with_logits(z,y,y)))
        self.assertEqual(terms[1:].sum().item(),0)

    def test_preparation_and_real_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); split=root/'split'; split.mkdir(); regions=root/'mat'; regions.mkdir()
            rows=[]
            for i in range(2):
                im=root/f'{i}.jpg'; target=root/f'{i}_t.png'; valid=root/f'{i}_v.png'
                Image.new('RGB',(8,8)).save(im)
                a=np.zeros((8,8),dtype=np.uint8); a[0,0]=255
                Image.fromarray(a).save(target); Image.fromarray(a).save(valid)
                rows.append([str(im),str(target),str(valid)])
                segs=np.empty((1,4),dtype=object)
                for j in range(4): segs[0,j]=np.tile(np.arange(8,dtype=np.uint8)//(4-j),(8,1))
                savemat(regions/f'{i}.mat',dict(segs=segs))
            (split/'train.json').write_text(json.dumps(rows[:1])); (split/'val.json').write_text(json.dumps(rows[1:]))
            out=root/'prepared'
            subprocess.run([sys.executable,str(ROOT/'tools/prepare_region_supervision.py'),
                '--split-source',str(split),'--regions',str(regions),'--output',str(out),'--size','16'],
                check=True,capture_output=True)
            from region_supervision_data import RegionSupervisionDataset
            sample=RegionSupervisionDataset(rows[:1],16,out/'targets')[0]
            self.assertEqual(tuple(sample['region_targets'].shape),(3,16,16))
            self.assertFalse((sample['region_masks']*sample['valid_mask']).any())
            self.assertFalse((out/'targets/1.npz').exists())
            self.assertEqual(json.loads((out/'preparation.json').read_text())['records'][0]['levels'],[1,2,3])


if __name__=='__main__': unittest.main()
