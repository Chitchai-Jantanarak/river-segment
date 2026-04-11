from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
import rasterio
from loguru import logger
from skimage.morphology import skeletonize

from ..domain.inference import ImageMetadata


def save_tif(band: np.ndarray, path: Path, meta: dict[str, Any]) -> Path:
    m = meta.copy()
    m.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)
    with rasterio.open(path, "w", **m) as dst:
        dst.write_band(1, band.astype(rasterio.uint8))
    return path


def save_png(draw_fn, path: Path, rgb: np.ndarray, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    ax.imshow(rgb)
    draw_fn(ax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def infer_centerline(
    mask: np.ndarray,
    meta: ImageMetadata,
    rgb: np.ndarray,
    out_dir: Union[Path, str],
    stem: str,
    skeleton: Optional[np.ndarray] = None,
) -> dict[str, Optional[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tif": None,
        "png": None,
    }

    if skeleton is None:
        skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)

    meta_dict = {
        "dtype": rasterio.uint8,
        "count": 1,
        "height": meta.height,
        "width": meta.width,
        "transform": meta.transform,
        "crs": meta.crs,
    }

    tif_path = out_dir / f"{stem}_river_centerline.tif"
    save_tif(skeleton, tif_path, meta_dict)
    result["tif"] = tif_path
    logger.info(f"  TIF  -> {tif_path}")

    png_path = out_dir / f"{stem}_river_centerline.png"

    def draw_fn(ax):
        ov = np.zeros((meta.height, meta.width, 4), dtype=np.float32)
        ov[mask > 0] = [0.05, 0.2, 0.9, 0.3]
        ax.imshow(ov)
        pts = np.argwhere(skeleton)
        if len(pts):
            ax.scatter(
                pts[:, 1],
                pts[:, 0],
                c="orange",
                s=0.5,
                linewidths=0,
                zorder=5,
            )

    save_png(draw_fn, png_path, rgb, stem)
    result["png"] = png_path
    logger.info(f"  PNG  -> {png_path}")

    return result
