from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from ..domain.inference import InferenceOptions
from ..domain.training import TrainingConfig


def load_base_config() -> dict[str, Any]:
    config_dir = Path(__file__).parent.parent.parent / "conf"
    base_path = config_dir / "config.yaml"
    if base_path.exists():
        return OmegaConf.load(base_path)
    return {}


def load_infer_config(overrides: Optional[dict[str, Any]] = None) -> InferenceOptions:
    cfg = OmegaConf.create(load_base_config())

    infer_path = Path(__file__).parent.parent.parent / "conf" / "infer.yaml"
    if infer_path.exists():
        infer_cfg = OmegaConf.load(infer_path)
        cfg = OmegaConf.merge(cfg, infer_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)

    return OmegaConf.to_object(cfg)


def load_train_config(overrides: Optional[dict[str, Any]] = None) -> TrainingConfig:
    cfg = OmegaConf.create(load_base_config())

    train_path = Path(__file__).parent.parent.parent / "conf" / "train.yaml"
    if train_path.exists():
        train_cfg = OmegaConf.load(train_path)
        cfg = OmegaConf.merge(cfg, train_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)

    return OmegaConf.to_object(cfg)


def load_finetune_config(overrides: Optional[dict[str, Any]] = None) -> TrainingConfig:
    cfg = OmegaConf.create(load_base_config())

    finetune_path = Path(__file__).parent.parent.parent / "conf" / "finetune.yaml"
    if finetune_path.exists():
        finetune_cfg = OmegaConf.load(finetune_path)
        cfg = OmegaConf.merge(cfg, finetune_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)

    return OmegaConf.to_object(cfg)


def infer_config_to_dict(opts: InferenceOptions) -> dict[str, Any]:
    return {
        "input": opts.input,
        "ckpt": opts.ckpt,
        "out": opts.out,
        "task": opts.task,
        "thresh": opts.thresh,
        "size": opts.size,
        "sword": opts.sword,
        "backbone": opts.backbone,
        "head": opts.head,
        "adaptor": opts.adaptor,
        "method": opts.method,
    }


def train_config_to_dict(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "data": cfg.data,
        "ckpt": cfg.ckpt,
        "out": cfg.out,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch": cfg.batch,
        "warmup": cfg.warmup,
        "phase": cfg.phase,
        "freeze_backbone": cfg.freeze_backbone,
    }
