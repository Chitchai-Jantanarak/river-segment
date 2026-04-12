from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
import rasterio
import torch
import torchvision.transforms as T
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

from src.data.controller import TIFFReader
from src.data.io import build_output_dir, find_tiff_files
from src.domain.inference import ImageMetadata
from src.models import load_checkpoint
from src.services.centerline import infer_centerline
from src.services.shape import infer_shape
from src.services.width import infer_width


def _preprocess(prob: np.ndarray, thresh: float) -> np.ndarray:
    mask = (prob > thresh).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            clean[lbl == i] = 1
    return clean


# Argument priority: CLI override  →  conf/infer.yaml defaults
@hydra.main(version_base=None, config_path="conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    if not cfg.input:
        logger.error("input is required: python main.py input=path/to/file.tif ckpt=path/to/model.pth")
        sys.exit(1)
    if not cfg.ckpt:
        logger.error("ckpt is required: python main.py input=... ckpt=path/to/model.pth")
        sys.exit(1)

    input_files = find_tiff_files(cfg.input)
    base_out = Path(cfg.out)
    base_out.mkdir(parents=True, exist_ok=True)

    logger.add(
        base_out / "infer.log",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.info(f"Loading checkpoint: {cfg.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(str(cfg.ckpt), cfg.size, cfg.backbone, cfg.head).to(device).eval()
    logger.info(f"Model ready on {device}")

    do_shape = cfg.task in ("all", "shape")
    do_centerline = cfg.task in ("all", "centerline")
    do_width = cfg.task in ("all", "width")
    sword_dir = Path(cfg.sword) if cfg.sword else None

    for tif_path in input_files:
        stem = tif_path.stem

        logger.info(f"Reading {tif_path}")
        with rasterio.open(tif_path) as src:
            raw = src.read()
            H, W = raw.shape[1], raw.shape[2]
            res_m = abs(src.transform.a)
            meta = ImageMetadata(shape=raw.shape, transform=src.transform, crs=src.crs, bounds=src.bounds, resolution=res_m)
        logger.info(f"  {W}x{H}px | {res_m:.2f}m/px | {raw.shape[0]} bands")

        reader = TIFFReader(tif_path)
        img = reader.normalize(raw, bands=4)
        rgb = reader.get_rgb(raw)

        inp = T.Compose([T.ToTensor(), T.Resize((cfg.size, cfg.size))])(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(inp, feat=True)[0]["water_mask"].squeeze().cpu().numpy()
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

        mask = _preprocess(prob, cfg.thresh)
        logger.info(f"  Water: {mask.mean() * 100:.2f}%  ({mask.sum() * res_m**2 / 1e6:.3f} km²)")

        skeleton: Optional[np.ndarray] = None
        if do_centerline or do_width:
            skeleton: np.ndarray = np.asarray(skeletonize(mask.astype(bool)), dtype=np.uint8)

        if do_shape:
            logger.info("--- shape ---")
            out_dir = build_output_dir(base_out, tif_path, "shape")
            out_dir.mkdir(parents=True, exist_ok=True)
            infer_shape(mask, meta, rgb, out_dir, stem)

        if do_centerline:
            logger.info("--- centerline ---")
            out_dir = build_output_dir(base_out, tif_path, "centerline")
            out_dir.mkdir(parents=True, exist_ok=True)
            infer_centerline(mask, meta, rgb, out_dir, stem, skeleton)

        if do_width:
            logger.info("--- width ---")
            out_dir = build_output_dir(base_out, tif_path, "width")
            out_dir.mkdir(parents=True, exist_ok=True)
            infer_width(mask, skeleton, meta, rgb, out_dir, stem, sword_dir)

        print(f"Done  ->  {build_output_dir(base_out, tif_path, cfg.task)}/")

    print(f"\nAll done  ->  {base_out}/")


def run_infer(
    input: str,
    ckpt: str,
    out: str = "results",
    task: str = "all",
    thresh: float = 0.4,
    size: int = 512,
    sword: Optional[str] = None,
    backbone: Optional[str] = None,
    head: Optional[str] = None,
) -> None:
    overrides = dict(input=input, ckpt=ckpt, out=out, task=task, thresh=thresh,
                     size=size, sword=sword, backbone=backbone, head=head)
    main(OmegaConf.create(overrides))


if __name__ == "__main__":
    main()
