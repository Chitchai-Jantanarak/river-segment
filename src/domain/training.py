from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrainingConfig:
    data: str
    ckpt: str
    out: str = "checkpoints"
    epochs: int = 30
    lr: float = 0.0001
    batch: int = 8
    warmup: int = 5
    phase: str = "train"
    freeze_backbone: bool = False

    @property
    def data_dir(self) -> Path:
        return Path(self.data)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.ckpt)

    @property
    def output_dir(self) -> Path:
        return Path(self.out)

    @property
    def best_checkpoint(self) -> Path:
        return self.output_dir / "best.pth.tar"

    @property
    def last_checkpoint(self) -> Path:
        return self.output_dir / "last.pth.tar"


@dataclass
class DatasetConfig:
    csv_path: str
    root: str
    size: int = 512
    augment: bool = True
    s2_idx: tuple[int, int, int, int] = (1, 2, 3, 7)

    @property
    def data_root(self) -> Path:
        return Path(self.root)

    @property
    def csv_file(self) -> Path:
        return Path(self.csv_path)


@dataclass
class CheckpointConfig:
    save_best: bool = True
    save_last: bool = True
    metric: str = "iou"
    mode: str = "max"


@dataclass
class TrainingResult:
    best_checkpoint: Optional[Path] = None
    last_checkpoint: Optional[Path] = None
    best_metric: float = 0.0
    final_metric: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": str(self.best_checkpoint) if self.best_checkpoint else None,
            "last": str(self.last_checkpoint) if self.last_checkpoint else None,
            "best_metric": self.best_metric,
            "final_metric": self.final_metric,
        }
