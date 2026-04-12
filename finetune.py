from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import S2Dataset
from src.models.factory import get_model
from src.models.loader import SATLAS_BACKBONES

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

EPS = 1e-7

_MODEL_CONFIGS: dict[str, SimpleNamespace] = {
    "resnet50": SimpleNamespace(
        segment_model="fpn",
        backbone="resnet50",
        head="no_head",
        adaptor="linear",
        method="vanilla",
        tasks=["water_mask"],
        task="water_mask",
        pretrained=1,
    ),
    "satlas_swinb": SimpleNamespace(
        segment_model="fpn",
        backbone="satlas_si_swinb",
        head="satlas_head",
        adaptor="linear",
        method="vanilla",
        tasks=["water_mask"],
        task="water_mask",
        pretrained=1,
    ),
}


class DiceBCE(nn.Module):
    def forward(self, prob: Tensor, y: Tensor) -> Tensor:
        inter = (prob * y).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (prob.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1)
        return F.binary_cross_entropy(prob, y) * 0.5 + dice.mean() * 0.5


def _iou(prob: Tensor, target: Tensor, thresh: float = 0.4) -> float:
    p = (prob > thresh).bool()
    t = target.bool()
    return ((p & t).sum() / ((p | t).sum() + EPS)).item()


def _freeze_encoder_only(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        is_adaptor = "adaptor" in name
        is_encoder = ("encoder" in name or "backbone.backbone" in name) and not is_adaptor
        param.requires_grad = not is_encoder


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _build_model(model_key: str, ckpt_path: str, size: int) -> nn.Module:
    cfg = _MODEL_CONFIGS[model_key]
    needs_pretrain = cfg.backbone in SATLAS_BACKBONES
    model = get_model(cfg, {"water_mask": 1}, num_inp_feats=4, pretrained=needs_pretrain)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd: dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd_clean, strict=False)
    miss, unex = len(msg.missing_keys), len(msg.unexpected_keys)
    logger.info(f"  Loaded checkpoint: missing={miss} unexpected={unex}")
    if model_key == "resnet50" and miss > 0:
        logger.info(f"  Note: adaptor Conv2d(4→3) randomly initialized — will be trained")
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
            lbls = lbls.unsqueeze(1).float().to(device)
            prob = model(imgs, feat=True)[0]["water_mask"]
            loss = loss_fn(prob, lbls)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_iou += _iou(prob.detach(), lbls) * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n, total_iou / n


def _save_bundle(model: nn.Module, epoch: int, val_iou: float, path: Path, model_key: str, size: int) -> None:
    cfg = _MODEL_CONFIGS[model_key]
    torch.save(
        {
            "epoch": epoch,
            "val_iou": val_iou,
            "state_dict": model.state_dict(),
            "opt": SimpleNamespace(
                segment_model=cfg.segment_model,
                backbone=cfg.backbone,
                head=cfg.head,
                adaptor=cfg.adaptor,
                method=cfg.method,
                tasks=cfg.tasks,
                task=cfg.task,
                pretrained=0,
                resize_size=size,
            ),
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune river segmentation model on S2 4-band data")
    p.add_argument("--model", required=True, choices=list(_MODEL_CONFIGS), help="Model config key")
    p.add_argument("--ckpt", required=True, help="Starting checkpoint (.pth / .pth.tar)")
    p.add_argument("--data", required=True, help="Dataset root (must contain train.csv + valid.csv)")
    p.add_argument("--out", default="model/finetune", help="Output directory (default: model/finetune)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5, help="Backbone-frozen warmup epochs")
    p.add_argument("--size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.add(out_dir / "train.log", rotation="00:00", retention="7 days",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}", level="DEBUG", enqueue=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  model_key: {args.model}")

    logger.info(f"Loading checkpoint: {args.ckpt}")
    model = _build_model(args.model, args.ckpt, args.size).to(device)
    loss_fn = DiceBCE()

    data_root = Path(args.data)
    tr_ds = S2Dataset(str(data_root / "train.csv"), str(data_root), args.size, augment=True)
    vl_ds = S2Dataset(str(data_root / "valid.csv"), str(data_root), args.size, augment=False)
    tr_dl = DataLoader(tr_ds, args.batch, shuffle=True, num_workers=4, pin_memory=True)
    vl_dl = DataLoader(vl_ds, args.batch, shuffle=False, num_workers=4, pin_memory=True)

    if args.warmup > 0:
        logger.info(f"Phase 1 — encoder frozen, adaptor trainable, {args.warmup} warmup epochs")
        _freeze_encoder_only(model)
        opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 5, weight_decay=1e-4)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, args.warmup)
        for ep in range(1, args.warmup + 1):
            tl, ti = _run_epoch(model, tr_dl, loss_fn, opt1, device)
            vl, vi = _run_epoch(model, vl_dl, loss_fn, None, device)
            sch1.step()
            logger.info(f"  warmup ep{ep}/{args.warmup}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")

    logger.info(f"Phase 2 — full model, {args.epochs} epochs")
    _unfreeze_all(model)
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, args.epochs)

    best_iou = 0.0
    best_path = out_dir / f"{args.model}_best.pth.tar"
    last_path = out_dir / f"{args.model}_last.pth.tar"

    for ep in range(1, args.epochs + 1):
        tl, ti = _run_epoch(model, tr_dl, loss_fn, opt2, device)
        vl, vi = _run_epoch(model, vl_dl, loss_fn, None, device)
        sch2.step()
        logger.info(f"ep {ep:03d}/{args.epochs}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")
        _save_bundle(model, ep, vi, last_path, args.model, args.size)
        if vi > best_iou:
            best_iou = vi
            _save_bundle(model, ep, vi, best_path, args.model, args.size)
            logger.info(f"  ★ best val_iou={best_iou:.4f}  → {best_path}")

    logger.info(f"Done.  Best val IoU = {best_iou:.4f}")
    logger.info(f"Use checkpoint: {best_path}")


if __name__ == "__main__":
    main()
