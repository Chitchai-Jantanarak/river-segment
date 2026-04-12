from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.models.factory import get_model

SATLAS_BACKBONES: frozenset[str] = frozenset({
    "satlas_si_swinb",
    "satlas_si_swint",
    "satlas_si_resnet50",
    "satlas_mi_swinb",
})


def load_checkpoint(
    ckpt_path: str,
    size: int = 512,
    backbone: Optional[str] = None,
    head: Optional[str] = None,
) -> nn.Module:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    is_bundle = isinstance(raw, dict) and "state_dict" in raw

    if is_bundle:
        opts: SimpleNamespace = raw["opt"]
        sd: dict = raw["state_dict"]
        logger.info(f"  Bundle: backbone={getattr(opts, 'backbone', '?')} head={getattr(opts, 'head', '?')}")
    else:
        sd = raw
        _backbone = backbone or "satlas_si_swinb"
        _head = head or "satlas_head"
        logger.info(f"  Plain .pth: backbone={_backbone} head={_head}")
        opts = SimpleNamespace(
            segment_model="fpn",
            backbone=_backbone,
            head=_head,
            adaptor="linear",
            method="vanilla",
            tasks=["water_mask"],
            task="water_mask",
            pretrained=1,
            resize_size=size,
        )

    needs_pretrain = getattr(opts, "backbone", "") in SATLAS_BACKBONES
    model = get_model(opts, {"water_mask": 1}, num_inp_feats=4, pretrained=needs_pretrain)

    sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd_clean, strict=is_bundle)
    if not is_bundle:
        miss, unex, tot = len(msg.missing_keys), len(msg.unexpected_keys), len(model.state_dict())
        logger.info(f"  Loaded: missing={miss} unexpected={unex} total={tot}")
        if (miss + unex) / max(tot, 1) > 0.15:
            raise RuntimeError(f"Checkpoint mismatch (missing={miss}, unexpected={unex}, total={tot})")

    return model
