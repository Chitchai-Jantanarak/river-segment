import sys
import os
from pathlib import Path

import numpy as np
import torch
from types import SimpleNamespace
import rasterio
import cv2
from scipy.ndimage import binary_fill_holes
import torchvision.transforms as T
from loguru import logger

from src.models import get_model
from src.data.controller import TIFFReader
from src.domain.inference import ImageMetadata
from src.services.shape import infer_shape

logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")


def find_tiff_files(input_path: str) -> list[Path]:
    """Find TIFF files - accepts file or directory input."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        tifs = list(input_path.rglob("*.tif"))
        if not tifs:
            tifs = list(input_path.rglob("*.tiff"))
        if not tifs:
            logger.error(f"No TIFF files found in {input_path}")
            sys.exit(1)
        logger.info(f"Found {len(tifs)} TIFF files in {input_path}")
        return tifs
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)


def main():
    input_path = sys.argv[2] if len(sys.argv) > 2 else "data/Hatyai_Full_Area_3m.tif"
    ckpt_path = sys.argv[4] if len(sys.argv) > 4 else "model/satlas/model.pth.tar"
    out_dir = sys.argv[6] if len(sys.argv) > 6 else "results"
    thresh = float(sys.argv[8]) if len(sys.argv) > 8 else 0.35
    size = int(sys.argv[10]) if len(sys.argv) > 10 else 512

    input_files = find_tiff_files(input_path)
    input_path = Path(input_files[0])
    input_dir = input_path.parent.name
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_bundle = isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt

    if is_bundle:
        ckpt_opt = raw_ckpt["opt"]
        sd = raw_ckpt["state_dict"]
        seg_model = getattr(ckpt_opt, "segment_model", "unet")
        backb = getattr(ckpt_opt, "backbone", "resnet50")
        logger.info(f"  Bundle -> segment: {seg_model}, backbone: {backb}")
    else:
        sd = raw_ckpt
        seg_model = "unet"
        backb = "resnet50"
        logger.info(f"  Plain .pth -> segment: unet")

    ckpt_opt_smp = SimpleNamespace(
        segment_model=seg_model,
        backbone=backb
        if backb
        in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x4d",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "mobilenet_v2",
        ]
        else "resnet50",
        resize_size=size,
    )
    try:
        model = get_model(ckpt_opt_smp, tasks_outputs={"water_mask": 1}, num_inp_feats=4, pretrained=False)
    except Exception:
        ckpt_opt_smp.segment_model = "unet"
        ckpt_opt_smp.backbone = "resnet50"
        model = get_model(ckpt_opt_smp, tasks_outputs={"water_mask": 1}, num_inp_feats=4, pretrained=False)

    sd_clean = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    load_msg = model.load_state_dict(sd_clean, strict=False)
    logger.info(f"  Loaded: missing={len(load_msg.missing_keys)}, unexpected={len(load_msg.unexpected_keys)}")
    model.eval()
    logger.info("Model ready")

    for input_path in input_files:
        stem = input_path.stem
        out_task_dir = out_dir / input_dir / "task" / stem
        out_task_dir.mkdir(parents=True, exist_ok=True)

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
        logger.info("--- river shape ---")

        meta = ImageMetadata(
            shape=raw.shape,
            transform=src.transform,
            crs=src.crs,
            bounds=src.bounds,
            resolution=res_m,
        )
        infer_shape(mask, meta, rgb, out_task_dir, stem)

        print(f"\nDone -> {out_task_dir}/")

    print(f"\nAll done -> {out_dir}/")


if __name__ == "__main__":
    main()
