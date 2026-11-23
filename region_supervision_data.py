from pathlib import Path
import numpy as np
import torch
from weak_data import WeakSalObjDataset


class RegionSupervisionDataset(WeakSalObjDataset):
    def __init__(self,samples,trainsize,regions):
        super().__init__(samples,trainsize)
        self.regions=Path(regions)

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        stem=self.samples[index][0].stem
        with np.load(self.regions/(stem+'.npz'),allow_pickle=False) as z:
            targets=z['targets']; masks=z['masks']
            shape=(3,self.trainsize,self.trainsize)
            if targets.shape!=shape or masks.shape!=shape:
                raise ValueError('Cached supervision shape mismatch: '+stem)
            if not np.isin(targets,[0,1]).all() or not np.isin(masks,[0,1]).all():
                raise ValueError('Cached supervision not binary: '+stem)
            sample['region_targets']=torch.from_numpy(targets.astype(np.float32))
            sample['region_masks']=torch.from_numpy(masks.astype(np.float32))
        if torch.any(sample['region_masks'] * sample['valid_mask']):
            raise ValueError('Expanded loss must not overwrite/double-count scribbles')
        return sample
