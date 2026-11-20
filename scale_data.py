"""Precomputed, aligned three-level maps; original scribble transforms retained."""
from pathlib import Path
import numpy as np
import torch
from weak_data import WeakSalObjDataset


class ScaleDataset(WeakSalObjDataset):
    def __init__(self, samples, trainsize, regions):
        super().__init__(samples, trainsize)
        self.regions = Path(regions)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        stem = self.samples[index][0].stem
        with np.load(self.regions / (stem + '.npz'), allow_pickle=False) as z:
            maps = z['maps']
            levels = z['actual_levels']
            if maps.shape != (3, self.trainsize, self.trainsize):
                raise ValueError('Prepared map size mismatch: ' + stem)
            if not np.issubdtype(maps.dtype, np.integer):
                raise ValueError('Integer maps required: ' + stem)
            sample['region_ids'] = torch.from_numpy(maps.astype(np.int64))
            sample['actual_levels'] = torch.from_numpy(levels.astype(np.int64))
        sample['id'] = stem
        return sample
