import os
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
import rasterio
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize
import torchvision.transforms as T
from loguru import logger
from tqdm import tqdm

from river_segment.domain.inference import (
    ImageMetadata,
)
from river_segment.data.controller import TIFFReader
from river_segment.data.loader import build_dataloader
from river_segment.services.shape import infer_shape
from river_segment.services.centerline import infer_centerline
from river_segment.services.width import infer_width


EPS = 1e-7


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


def load_model(ckpt_path: str, size: int, backbone: Optional[str], head: Optional[str]):
    from river_segment.models import get_model

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_bundle = isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt

    if is_bundle:
        ckpt_opt = raw_ckpt["opt"]
        sd = raw_ckpt["state_dict"]
        logger.info(
            f"  Bundle -> backbone: {getattr(ckpt_opt, 'backbone', '?')}  head: {getattr(ckpt_opt, 'head', '?')}"
        )
    else:
        sd = raw_ckpt
        backbone = backbone or "satlas_si_swinb"
        head = head or "satlas_head"
        logger.info(f"  Plain .pth -> backbone: {backbone}  head: {head}")
        ckpt_opt = SimpleNamespace(
            segment_model="fpn",
            backbone=backbone,
            head=head,
            adaptor="linear",
            method="vanilla",
            tasks=["water_mask"],
            task="water_mask",
            pretrained=0,
            resize_size=size,
        )

    model = get_model(ckpt_opt, tasks_outputs={"water_mask": 1}, num_inp_feats=4, pretrained=False)
    sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
    load_msg = model.load_state_dict(sd_clean, strict=is_bundle)
    if not is_bundle:
        miss = len(load_msg.missing_keys)
        unex = len(load_msg.unexpected_keys)
        tot = len(model.state_dict())
        if (miss + unex) / max(tot, 1) > 0.15:
            raise RuntimeError(f"Checkpoint mismatch (missing={miss}, unexpected={unex}, total={tot}).")
    model.cuda().eval()
    return model


def run_model(model, inp: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        prob = torch.sigmoid(model(inp, feat=True)[0]["water_mask"])
    return prob.squeeze().cpu().numpy()


def preprocess_mask(prob: np.ndarray, thresh: float, H: int, W: int) -> np.ndarray:
    mask = (prob > thresh).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)

    n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_cc):
        if stats_cc[i, cv2.CC_STAT_AREA] >= 50:
            clean[lbl_cc == i] = 1
    return clean


@hydra.main(version_base=None, config_path="conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    input_files = find_tiff_files(cfg.input)
    input_path = Path(input_files[0])
    input_dir = input_path.parent.name

    cfg.out = Path(cfg.out)
    os.makedirs(cfg.out, exist_ok=True)
    logger.add(
        cfg.out / "infer.log",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.info(f"Loading checkpoint: {cfg.ckpt}")
    model = load_model(cfg.ckpt, cfg.size, cfg.backbone, cfg.head)
    logger.info("Model ready")

    do_shape = cfg.task in ("all", "shape")
    do_centerline = cfg.task in ("all", "centerline")
    do_width = cfg.task in ("all", "width")

    for input_path in input_files:
        stem = input_path.stem
        out_dir = cfg.out / input_dir / "task" / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Reading {input_path} ...")
        with rasterio.open(input_path) as src:
            raw = src.read()
            H, W = raw.shape[1], raw.shape[2]
            res_m = abs(src.transform.a)
            logger.info(f"  {W}x{H} px | {res_m:.2f} m/px | {raw.shape[0]} bands")

            meta = ImageMetadata(
                shape=raw.shape,
                transform=src.transform,
                crs=src.crs,
                bounds=src.bounds,
                resolution=res_m,
            )

        reader = TIFFReader(input_path)
        img = reader.normalize(raw, bands=4)
        rgb = reader.get_rgb(raw)

        inp = T.Compose([T.ToTensor(), T.Resize((cfg.size, cfg.size))])(img)
        inp = inp.unsqueeze(0).cuda()

        prob = run_model(model, inp)
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

        mask = preprocess_mask(prob, cfg.thresh, H, W)
        logger.info(f"  Water: {mask.mean() * 100:.2f}%  ({mask.sum() * res_m**2 / 1e6:.3f} km²)")

        skeleton = None
        if cfg.task in ("all", "centerline") or cfg.task in ("all", "width"):
            skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)

        result = {
            "shape": {},
            "centerline": {},
            "width": {},
        }

        if do_shape:
            logger.info("--- river shape ---")
            result["shape"] = infer_shape(mask, meta, rgb, out_dir, stem)

        if do_centerline:
            logger.info("--- river centerline ---")
            result["centerline"] = infer_centerline(mask, meta, rgb, out_dir, stem, skeleton)

        if do_width:
            logger.info("--- river width ---")
            sword_dir = Path(cfg.sword) if cfg.sword else None
            result["width"] = infer_width(mask, skeleton, meta, rgb, out_dir, stem, sword_dir)

        print(f"\nDone  ->  {out_dir}/")
        if do_shape:
            print(f"  {stem}_river_shape.tif")
        if do_centerline:
            print(f"  {stem}_river_centerline.tif")
        if do_width:
            print(f"  {stem}_width_numbers.csv")

    print(f"\nAll done  ->  {cfg.out}/")


def run_infer(
    input: str,
    ckpt: str,
    out: str = "results",
    task: str = "all",
    thresh: float = 0.35,
    size: int = 512,
    sword: Optional[str] = None,
    backbone: Optional[str] = None,
    head: Optional[str] = None,
    adaptor: str = "linear",
    method: str = "vanilla",
) -> dict[str, Any]:
    overrides = {
        "input": input,
        "ckpt": ckpt,
        "out": out,
        "task": task,
        "thresh": thresh,
        "size": size,
        "sword": sword,
        "backbone": backbone,
        "head": head,
        "adaptor": adaptor,
        "method": method,
    }
    cfg = OmegaConf.create(overrides)
    main(cfg)


class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        inter = (p * y).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1)
        return 0.5 * self.bce(logits, y) + 0.5 * dice.mean()


def set_backbone_grad(model: torch.nn.Module, requires_grad: bool):
    for name, param in model.named_parameters():
        if "backbone" in name or "encoder" in name:
            param.requires_grad = requires_grad


def iou_metric(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    p = (torch.sigmoid(pred_logits) > 0.4).bool()
    t = target.bool()
    return ((p & t).sum() / ((p | t).sum() + EPS)).item()


def run_one_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_iou = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, lbls in tqdm(loader, leave=False, desc="  batch"):
            imgs = imgs.to(device)
            lbls = lbls.unsqueeze(1).to(device)
            logits = model(imgs, feat=True)[0]["water_mask"]
            loss = loss_fn(logits, lbls)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_iou += iou_metric(logits, lbls) * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n, total_iou / n


def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    val_iou: float,
    path: Path,
    size: int,
):
    torch.save(
        {
            "epoch": epoch,
            "val_iou": val_iou,
            "state_dict": model.state_dict(),
            "opt": SimpleNamespace(
                segment_model="fpn",
                backbone="satlas_si_swinb",
                head="satlas_head",
                adaptor="linear",
                method="vanilla",
                tasks=["water_mask"],
                task="water_mask",
                pretrained=0,
                resize_size=size,
            ),
        },
        path,
    )


def run_train(
    data: str,
    ckpt: str,
    out: str = "model",
    epochs: int = 30,
    lr: float = 1e-4,
    batch: int = 8,
    warmup: int = 5,
    size: int = 512,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading checkpoint: {ckpt}")
    model = load_model(ckpt, size, None, None).to(device)
    loss_fn = DiceBCE()

    data_dir = Path(data)
    tr_loader = build_dataloader(str(data_dir / "train.csv"), data, size, batch, True, True)
    vl_loader = build_dataloader(str(data_dir / "valid.csv"), data, size, batch, False, False)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if warmup > 0:
        logger.info(f"Phase 1 -- backbone frozen, {warmup} warmup epochs")
        set_backbone_grad(model, False)
        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * 5,
            weight_decay=1e-4,
        )
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, warmup)
        for ep in range(1, warmup + 1):
            tl, ti = run_one_epoch(model, tr_loader, loss_fn, opt1, device)
            vl, vi = run_one_epoch(model, vl_loader, loss_fn, None, device)
            sch1.step()
            logger.info(f"  warmup ep{ep}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")

    logger.info(f"Phase 2 -- full model, {epochs} epochs")
    set_backbone_grad(model, True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, epochs)

    best_iou = 0.0
    best_path = out_dir / "s2_best.pth.tar"
    last_path = out_dir / "s2_last.pth.tar"

    for ep in range(1, epochs + 1):
        tl, ti = run_one_epoch(model, tr_loader, loss_fn, opt2, device)
        vl, vi = run_one_epoch(model, vl_loader, loss_fn, None, device)
        sch2.step()
        logger.info(f"ep {ep:03d}/{epochs}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")
        save_checkpoint(model, ep, vi, last_path, size)
        if vi > best_iou:
            best_iou = vi
            save_checkpoint(model, ep, vi, best_path, size)
            logger.info(f" - new best val_iou={best_iou:.4f}  → {best_path}")

    logger.info(f"Done.  Best val IoU = {best_iou:.4f}")
    logger.info("Use this checkpoint in infer:")
    logger.info(f"  --ckpt {best_path}")

    return str(best_path)


if __name__ == "__main__":
    main()
