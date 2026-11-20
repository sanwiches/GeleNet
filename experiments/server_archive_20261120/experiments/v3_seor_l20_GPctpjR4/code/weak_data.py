"""Dataset for explicitly paired, already decoded binary scribble masks.

No dataset-specific directory layout or raw scribble encoding is inferred here.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WeakSalObjDataset(Dataset):
    """Each sample is (image_path, target_path, valid_mask_path).

    The two masks must be single-channel images with pixel values 0 or 255.
    target=255 means foreground; valid_mask=255 means annotated. Target values
    outside the valid mask are ignored by the loss.
    """

    def __init__(self, samples, trainsize):
        if not samples:
            raise ValueError("samples must contain at least one triplet")
        if trainsize <= 0:
            raise ValueError("trainsize must be positive")
        self.samples = [tuple(map(Path, sample)) for sample in samples]
        if any(len(sample) != 3 for sample in self.samples):
            raise ValueError("each sample must contain image, target and valid_mask paths")
        self.trainsize = trainsize
        self.image_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _binary_mask(path, size):
        with Image.open(path) as source:
            if source.mode != "L" or source.size != size:
                raise ValueError("mask must be single-channel and match image size: {}".format(path))
            pixels = np.asarray(source)
            if not np.all((pixels == 0) | (pixels == 255)):
                raise ValueError("mask must contain only 0 and 255: {}".format(path))
            mask = source.resize(size, Image.NEAREST)
            return mask.copy()

    def __getitem__(self, index):
        image_path, target_path, valid_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        target = self._binary_mask(target_path, image.size)
        valid_mask = self._binary_mask(valid_path, image.size)
        shape = (self.trainsize, self.trainsize)
        target = target.resize(shape, Image.NEAREST)
        valid_mask = valid_mask.resize(shape, Image.NEAREST)
        return {
            "image": self.image_transform(image),
            "target": torch.from_numpy(np.array(target, dtype=np.float32)).unsqueeze(0) / 255.0,
            "valid_mask": torch.from_numpy(np.array(valid_mask, dtype=np.float32)).unsqueeze(0) / 255.0,
        }


def get_weak_loader(samples, batchsize, trainsize, shuffle=True, num_workers=0,
                    pin_memory=False):
    return DataLoader(WeakSalObjDataset(samples, trainsize), batch_size=batchsize,
                      shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
