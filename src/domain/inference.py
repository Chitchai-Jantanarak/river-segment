from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class InferenceOptions:
    input: str
    ckpt: str
    out: str = "results"
    task: str = "all"
    thresh: float = 0.35
    size: int = 512
    sword: Optional[str] = None
    backbone: Optional[str] = None
    head: Optional[str] = None
    adaptor: str = "linear"
    method: str = "vanilla"


@dataclass
class ImageMetadata:
    shape: tuple[int, int, int]
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    resolution: float

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]

    @property
    def bands(self) -> int:
        return self.shape[0]


@dataclass
class InferenceResult:
    shape_path: Optional[Path] = None
    shape_png: Optional[Path] = None
    shape_gpkg: Optional[Path] = None
    centerline_path: Optional[Path] = None
    centerline_png: Optional[Path] = None
    width_csv: Optional[Path] = None
    width_png: Optional[Path] = None
    width_stats: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": {
                "tif": str(self.shape_path) if self.shape_path else None,
                "png": str(self.shape_png) if self.shape_png else None,
                "gpkg": str(self.shape_gpkg) if self.shape_gpkg else None,
            },
            "centerline": {
                "tif": str(self.centerline_path) if self.centerline_path else None,
                "png": str(self.centerline_png) if self.centerline_png else None,
            },
            "width": {
                "csv": str(self.width_csv) if self.width_csv else None,
                "png": str(self.width_png) if self.width_png else None,
                "stats": self.width_stats,
            },
        }
