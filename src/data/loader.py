import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from loguru import logger

EPS = 1e-7
S2_IDX = (1, 2, 3, 7)


class S2Dataset(Dataset):
    def __init__(self, csv_path: str, root: str, size: int = 512, augment: bool = True):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                s2 = self.root / row["s2_path_reprojected"]
                lbl = self.root / row["label_path"]
                if s2.exists() and lbl.exists():
                    self.samples.append((str(s2), str(lbl)))
        logger.info(f"  {len(self.samples)} samples <- {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s2p, lblp = self.samples[i]

        with rasterio.open(s2p) as src:
            raw = src.read()
        raw = raw[S2_IDX] if raw.shape[0] > max(S2_IDX) else raw[:4]
        raw = raw.astype(np.float32)
        lo, hi = np.percentile(raw, 2), np.percentile(raw, 98)
        raw = np.clip((raw - lo) / max(hi - lo, EPS), 0, 1)
        img = torch.from_numpy(raw)

        with rasterio.open(lblp) as src:
            lbl = src.read(1).astype(np.float32)
        lbl = torch.from_numpy(lbl).unsqueeze(0)

        img = TF.resize(img, [self.size, self.size], antialias=True)
        lbl = TF.resize(lbl, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, lbl = TF.hflip(img), TF.hflip(lbl)
            if random.random() > 0.5:
                img, lbl = TF.vflip(img), TF.vflip(lbl)
            k = random.randint(0, 3)
            img = torch.rot90(img, k, [1, 2])
            lbl = torch.rot90(lbl, k, [1, 2])

        return img, lbl.squeeze(0)


def build_dataloader(
    csv_path: str,
    root: str,
    size: int = 512,
    batch: int = 8,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = S2Dataset(csv_path, root, size, augment)
    return DataLoader(
        dataset,
        batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_loaders(
    data_dir: str,
    train_csv: str = "train.csv",
    valid_csv: str = "valid.csv",
    test_csv: Optional[str] = None,
    size: int = 512,
    batch: int = 8,
    augment: bool = True,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    root = Path(data_dir)
    tr_loader = build_dataloader(str(root / train_csv), data_dir, size, batch, augment, True, num_workers)
    vl_loader = build_dataloader(str(root / valid_csv), data_dir, size, batch, False, False, num_workers)
    te_loader = None
    if test_csv:
        te_loader = build_dataloader(str(root / test_csv), data_dir, size, batch, False, False, num_workers)
    return tr_loader, vl_loader, te_loader
