import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torchvision.transforms as T
from loguru import logger
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize

from src.data.controller import TIFFReader
from src.data.io import build_output_dir, find_tiff_files
from src.domain.inference import ImageMetadata
from src.models import load_checkpoint
from src.services.width import infer_width

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="River width measurement inference")
    p.add_argument("-i", "--input", required=True, help="TIFF file or directory")
    p.add_argument("-c", "--ckpt", required=True, help="Checkpoint (.pth / .pth.tar)")
    p.add_argument("-o", "--out", default="results", help="Output root (default: results)")
    p.add_argument("-t", "--thresh", type=float, default=0.4, help="Water threshold (default: 0.4)")
    p.add_argument("-s", "--size", type=int, default=512, help="Model input size (default: 512)")
    p.add_argument("--backbone", default=None, help="Backbone override for plain .pth")
    p.add_argument("--head", default=None, help="Head override for plain .pth")
    p.add_argument("--sword", default=None, help="SWORD shapefile directory (optional)")
    return p.parse_args()


def _preprocess(prob: np.ndarray, thresh: float) -> np.ndarray:
    mask = (prob > thresh).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            clean[lbl == i] = 1
    return clean


def main() -> None:
    args = _parse_args()
    input_files = find_tiff_files(args.input)
    base_out = Path(args.out)
    sword_dir = Path(args.sword) if args.sword else None

    logger.info(f"Loading checkpoint: {args.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt, args.size, args.backbone, args.head).to(device).eval()
    logger.info(f"Model ready on {device}")

    for tif_path in input_files:
        stem = tif_path.stem
        out_dir = build_output_dir(base_out, tif_path, "width")
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {tif_path}")
        with rasterio.open(tif_path) as src:
            raw = src.read()
            H, W = raw.shape[1], raw.shape[2]
            res_m = abs(src.transform.a)
            meta = ImageMetadata(shape=raw.shape, transform=src.transform, crs=src.crs, bounds=src.bounds, resolution=res_m)
        logger.info(f"  {W}x{H}px | {res_m:.2f}m/px | {raw.shape[0]} bands")

        reader = TIFFReader(tif_path)
        img = reader.normalize(raw, bands=4)
        rgb = reader.get_rgb(raw)

        inp = T.Compose([T.ToTensor(), T.Resize((args.size, args.size))])(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(inp, feat=True)[0]["water_mask"].squeeze().cpu().numpy()
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

        mask = _preprocess(prob, args.thresh)
        skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
        logger.info(f"  Water: {mask.mean() * 100:.2f}%")

        infer_width(mask, skeleton, meta, rgb, out_dir, stem, sword_dir)
        print(f"Done  ->  {out_dir}/")

    print(f"\nAll done  ->  {base_out}/")


if __name__ == "__main__":
    main()
