import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from types import SimpleNamespace

from river_segment.data.loader import S2Dataset


EPS = 1e-7


class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        inter = (p * y).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1)
        return 0.5 * self.bce(logits, y) + 0.5 * dice.mean()


def build_model(ckpt_path: Path, size: int):
    from models.get_model import get_model

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        ns, sd = raw["opt"], raw["state_dict"]
    else:
        sd = raw
        ns = SimpleNamespace(
            segment_model="fpn",
            backbone="satlas_si_swinb",
            head="satlas_head",
            adaptor="linear",
            method="vanilla",
            tasks=["water_mask"],
            task="water_mask",
            pretrained=1,
            resize_size=size,
        )
    model = get_model(ns, tasks_outputs={"water_mask": 1}, num_inp_feats=4, pretrained=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def set_backbone_grad(model, requires_grad: bool):
    for name, param in model.named_parameters():
        if "backbone" in name or "encoder" in name:
            param.requires_grad = requires_grad


def iou(pred_logits, target):
    p = (torch.sigmoid(pred_logits) > 0.4).bool()
    t = target.bool()
    return ((p & t).sum() / ((p | t).sum() + EPS)).item()


def run_one_epoch(model, loader, loss_fn, optimizer, device):
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
            total_iou += iou(logits, lbls) * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n, total_iou / n


def save(model, epoch, val_iou, path, size):
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


@hydra.main(version_base=None, config_path="config/hydra", config_name="train")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out, exist_ok=True)
    logger.add(
        cfg.out / "train.log",
        rotation="00:00",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        enqueue=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_model(Path(cfg.ckpt), cfg.size).to(device)
    loss_fn = DiceBCE()

    train_csv = Path(cfg.data) / "train.csv"
    valid_csv = Path(cfg.data) / "valid.csv"
    tr_ds = S2Dataset(train_csv, Path(cfg.data), cfg.size, augment=True)
    vl_ds = S2Dataset(valid_csv, Path(cfg.data), cfg.size, augment=False)
    tr_dl = DataLoader(tr_ds, cfg.batch, shuffle=True, num_workers=4, pin_memory=True)
    vl_dl = DataLoader(vl_ds, cfg.batch, shuffle=False, num_workers=4, pin_memory=True)

    if cfg.warmup > 0:
        logger.info(f"Phase 1 — backbone frozen, {cfg.warmup} warmup epochs")
        set_backbone_grad(model, False)
        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr * 5,
            weight_decay=1e-4,
        )
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, cfg.warmup)
        for ep in range(1, cfg.warmup + 1):
            tl, ti = run_one_epoch(model, tr_dl, loss_fn, opt1, device)
            vl, vi = run_one_epoch(model, vl_dl, loss_fn, None, device)
            sch1.step()
            logger.info(f"  warmup ep{ep}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")

    logger.info(f"Phase 2 — full model, {cfg.epochs} epochs")
    set_backbone_grad(model, True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, cfg.epochs)

    best_iou = 0.0
    best_path = Path(cfg.out) / "s2_best.pth.tar"
    last_path = Path(cfg.out) / "s2_last.pth.tar"

    for ep in range(1, cfg.epochs + 1):
        tl, ti = run_one_epoch(model, tr_dl, loss_fn, opt2, device)
        vl, vi = run_one_epoch(model, vl_dl, loss_fn, None, device)
        sch2.step()
        logger.info(f"ep {ep:03d}/{cfg.epochs}  tr_iou={ti:.3f} loss={tl:.4f} | val_iou={vi:.3f} loss={vl:.4f}")
        save(model, ep, vi, last_path, cfg.size)
        if vi > best_iou:
            best_iou = vi
            save(model, ep, vi, best_path, cfg.size)
            logger.info(f"  ★ new best val_iou={best_iou:.4f}  → {best_path}")

    logger.info(f"Done.  Best val IoU = {best_iou:.4f}")
    logger.info("Use this checkpoint in infer:")
    logger.info(f"  --ckpt {best_path}")


def run_train(
    data_dir: str,
    ckpt: str,
    out: str = "model",
    epochs: int = 30,
    lr: float = 1e-4,
    batch: int = 8,
    size: int = 512,
    warmup: int = 5,
) -> None:
    overrides = {
        "data": data_dir,
        "ckpt": ckpt,
        "out": out,
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "size": size,
        "warmup": warmup,
    }
    cfg = OmegaConf.create(overrides)
    main(cfg)


if __name__ == "__main__":
    main()
