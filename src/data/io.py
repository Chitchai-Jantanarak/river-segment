import sys
from pathlib import Path

import numpy as np
from loguru import logger


def find_tiff_files(input_path: str | Path) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        tifs = sorted(p.rglob("*.tif")) or sorted(p.rglob("*.tiff"))
        if not tifs:
            logger.error(f"No TIFF files found in {p}")
            sys.exit(1)
        logger.info(f"Found {len(tifs)} TIFF file(s) in {p}")
        return tifs
    logger.error(f"Input path not found: {p}")
    sys.exit(1)


def build_output_dir(base_out: Path, input_path: Path, task: str) -> Path:
    return base_out / input_path.parent.name / task / input_path.stem


def pad_bands(raw: np.ndarray, target: int = 4) -> np.ndarray:
    if raw.shape[0] >= target:
        return raw[:target].astype(np.float32)
    r = raw.astype(np.float32)
    synthetic = r.mean(axis=0, keepdims=True)
    extra = target - r.shape[0]
    return np.concatenate([r] + [synthetic] * extra, axis=0)
