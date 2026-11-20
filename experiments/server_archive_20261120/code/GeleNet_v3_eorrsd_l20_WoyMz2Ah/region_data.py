
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from weak_data import WeakSalObjDataset

ROOT = Path(__file__).resolve().parent / "regions_l20"

class RegionDataset(WeakSalObjDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = self.samples[index][0]
        a = np.load(ROOT / (image.stem + ".npy"), allow_pickle=False)
        with Image.open(image) as im:
            if a.shape != (im.height, im.width):
                raise ValueError("Region size mismatch: " + image.stem)
        a = np.array(Image.fromarray(a.astype(np.int32)).resize(
            (self.trainsize, self.trainsize), Image.NEAREST),
            dtype=np.int64)
        sample["region_ids"] = torch.from_numpy(a)
        return sample

def get_weak_loader(samples, batchsize, trainsize, shuffle=True,
                    num_workers=0, pin_memory=False):
    return DataLoader(RegionDataset(samples, trainsize),
                      batch_size=batchsize, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
