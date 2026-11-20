"""Isolated actual KTM classes plus synthetic MAT preparation, no pretrained backbone."""
import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import savemat
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class IntegrationTests(unittest.TestCase):
    def test_actual_ktm_fixed20_and_adaptive_backward(self):
        # Extract the real fixed-level KTM and its convolution dependency only.
        # The backbone is intentionally not constructed in this isolated test.
        source=ast.parse((ROOT/'model/GeleNet_models.py').read_text(encoding='utf-8'))
        source.body=[n for n in source.body if isinstance(n,ast.ClassDef)
                     and n.name in ('BasicConv2d','KTM')]
        ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,Softmax=torch.nn.Softmax)
        exec(compile(source,'actual_fixed_level_ktm','exec'),ns)
        module=types.ModuleType('model.GeleNet_models')
        module.KTM=ns['KTM']; module.GeleNet=type('UnusedBackboneBase',(),{})
        old=sys.modules.get('model.GeleNet_models')
        sys.modules['model.GeleNet_models']=module
        try:
            spec=importlib.util.spec_from_file_location('isolated_adaptive',ROOT/'model/adaptive_GeleNet.py')
            adaptive=importlib.util.module_from_spec(spec); spec.loader.exec_module(adaptive)
            torch.manual_seed(42)
            baseline=module.KTM().eval()
            fixed=adaptive.ScaleKTM(mode='fixed20').eval()
            fixed.load_state_dict(baseline.state_dict(),strict=False)
            # Nonzero gamma ensures a broken bias path cannot pass vacuously.
            for m in [baseline,fixed]:
                m.gamma_2.data.fill_(1); m.gamma_3.data.fill_(1)
            x=torch.randn(2,32,4,4); y=torch.randn_like(x)
            ids=torch.randint(0,5,(2,3,8,8))
            self.assertTrue(torch.allclose(baseline(x,y,ids[:,1]),fixed(x,y,ids),atol=1e-6))
            learned=adaptive.ScaleKTM(mode='adaptive').eval()
            learned.load_state_dict(fixed.state_dict())
            loss=learned(x,y,ids).square().mean(); loss.backward()
            grad=learned.scale_prior.gate[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
        finally:
            if old is None: sys.modules.pop('model.GeleNet_models',None)
            else: sys.modules['model.GeleNet_models']=old

    def test_preparation_split_and_short_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); regions=root/'mat'; regions.mkdir(); entries=[]
            for i in range(5):
                image=root/(str(i)+'.png'); mask=root/(str(i)+'_mask.png')
                Image.new('RGB',(8,8)).save(image); Image.new('L',(8,8)).save(mask)
                entries.append([str(image),str(mask),str(mask)])
                segs=np.empty((1,2),dtype=object)
                segs[0,0]=np.ones((8,8),dtype=np.uint8)
                segs[0,1]=np.tile(np.arange(8,dtype=np.uint8),(8,1))
                savemat(regions/(str(i)+'.mat'),dict(segs=segs))
            mf=root/'manifest.json'; mf.write_text(json.dumps(entries))
            output=root/'prepared'
            subprocess.run([sys.executable,str(ROOT/'tools/prepare_scale_experiment.py'),
                '--manifest',str(mf),'--regions',str(regions),'--output',str(output),
                '--val-count','2','--size','16'],check=True,capture_output=True)
            train=json.loads((output/'train.json').read_text()); val=json.loads((output/'val.json').read_text())
            self.assertEqual((len(train),len(val)),(3,2))
            self.assertFalse({r[0] for r in train}&{r[0] for r in val})
            with np.load(output/'regions/0.npz') as z:
                self.assertEqual(z['maps'].shape,(3,16,16))
                self.assertEqual(z['actual_levels'].tolist(),[2,2,2])
                self.assertTrue(np.array_equal(z['maps'][0],z['maps'][2]))
            from scale_data import ScaleDataset
            ds=ScaleDataset(train,16,output/'regions')
            sample=ds[0]
            self.assertEqual(tuple(sample['region_ids'].shape),(3,16,16))
            self.assertEqual(tuple(sample['image'].shape),(3,16,16))
            self.assertEqual(sample['region_ids'].dtype,torch.int64)


if __name__=='__main__': unittest.main()
