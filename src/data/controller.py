from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from torch.utils.data import DataLoader

from .loader import S2Dataset, build_loaders


class DataController:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_s2(self, csv_path: str) -> S2Dataset:
        return S2Dataset(csv_path, str(self.data_dir), size=512, augment=False)

    def load_labels(self, csv_path: str) -> S2Dataset:
        return S2Dataset(csv_path, str(self.data_dir), size=512, augment=False)

    def build_loaders(
        self,
        train_csv: str = "train.csv",
        valid_csv: str = "valid.csv",
        test_csv: Optional[str] = None,
        size: int = 512,
        batch: int = 8,
        augment: bool = True,
        num_workers: int = 4,
    ) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        return build_loaders(
            str(self.data_dir),
            train_csv,
            valid_csv,
            test_csv,
            size,
            batch,
            augment,
            num_workers,
        )


class TIFFReader:
    def __init__(self, path: str):
        self.path = Path(path)

    def read(self) -> tuple[np.ndarray, dict[str, Any]]:
        with rasterio.open(self.path) as src:
            raw = src.read()
            meta = {
                "shape": raw.shape,
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "resolution": abs(src.transform.a),
            }
        return raw, meta

    def normalize(self, raw: np.ndarray, bands: int = 4) -> np.ndarray:
        EPS = 1e-7
        if raw.shape[0] >= bands:
            img = raw[:bands].astype(np.float32)
        elif raw.shape[0] == 3:
            r = raw.astype(np.float32)
            img = np.concatenate([r, r.mean(axis=0, keepdims=True)], axis=0)
        else:
            raise ValueError(f"Need >= 3 bands, got {raw.shape[0]}")

        img = np.transpose(img, (1, 2, 0))
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        img = np.clip((img - lo) / max(hi - lo, EPS), 0, 1).astype(np.float32)
        return img

    def get_rgb(self, raw: np.ndarray) -> np.ndarray:
        EPS = 1e-7
        rgb_raw = raw[:3].astype(np.float32)
        lo, hi = np.percentile(rgb_raw, 2), np.percentile(rgb_raw, 98)
        rgb = np.clip((rgb_raw - lo) / max(hi - lo, EPS), 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
        return rgb
