import sys
from loguru import logger

from .domain import InferenceOptions, ImageMetadata, InferenceResult
from .domain import TrainingConfig, DatasetConfig, CheckpointConfig
from .data import DataController, S2Dataset
from .services import infer_shape, infer_centerline, infer_width

__version__ = "0.1.0"

logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            "level": "DEBUG",
        },
    ],
)

__all__ = [
    "InferenceOptions",
    "ImageMetadata",
    "InferenceResult",
    "TrainingConfig",
    "DatasetConfig",
    "CheckpointConfig",
    "DataController",
    "S2Dataset",
    "infer_shape",
    "infer_centerline",
    "infer_width",
    "run_infer",
    "train",
]
