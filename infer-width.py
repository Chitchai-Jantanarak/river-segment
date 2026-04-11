import sys
import os
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace
import rasterio
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize
import torchvision.transforms as T
from loguru import logger

from river_segment.models import get_model
from river_segment.data.controller import TIFFReader
from river_segment.domain.inference import ImageMetadata
from river_segment.services.width import infer_width

logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")


def main():
    input_path = sys.argv[2] if len(sys.argv) > 2 else "data/Hatyai_Full_Area_3m.tif"
    ckpt_path = sys.argv[4] if len(sys.argv) > 4 else "model/satlas/model.pth.tar"
    out_dir = sys.argv[6] if len(sys.argv) > 6 else "results"
    thresh = float(sys.argv[8]) if len(sys.argv) > 8 else 0.35
    size = int(sys.argv[10]) if len(sys.argv) > 10 else 512

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt_opt = SimpleNamespace(
        segment_model="unet",
        backbone="resnet50",
        head="unet",
        resize_size=size,
    )
    model = get_model(ckpt_opt, tasks_outputs={"water_mask": 1}, num_inp_feats=4, pretrained=False)

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw_ckpt.get("state_dict", raw_ckpt)
    sd_clean = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd_clean, strict=False)
    model.eval()
    logger.info("Model ready")

    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    stem = input_path.stem
    out_dir = Path(out_dir)

    logger.info(f"Reading {input_path} ...")
    with rasterio.open(input_path) as src:
        raw = src.read()
        H, W = raw.shape[1], raw.shape[2]
        res_m = abs(src.transform.a)
        logger.info(f"  {W}x{H} px | {res_m:.2f} m/px | {raw.shape[0]} bands")

    reader = TIFFReader(input_path)
    img = reader.normalize(raw, bands=4)
    rgb = reader.get_rgb(raw)

    img_tensor = T.Compose([T.ToTensor(), T.Resize((size, size))])(img)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        out = model(img_tensor)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()

    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

    mask = (prob > thresh).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)

    n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_cc):
        if stats_cc[i, cv2.CC_STAT_AREA] >= 50:
            clean[lbl_cc == i] = 1
    mask = clean

    logger.info(f"  Water: {mask.mean() * 100:.2f}%  ({mask.sum() * res_m**2 / 1e6:.3f} km²)")
    logger.info("Computing skeleton...")

    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)

    logger.info("--- river width ---")

    meta = ImageMetadata(
        shape=raw.shape,
        transform=src.transform,
        crs=src.crs,
        bounds=src.bounds,
        resolution=res_m,
    )
    infer_width(mask, skeleton, meta, rgb, out_dir, stem)

    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
