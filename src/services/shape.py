from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from loguru import logger

from ..domain.inference import ImageMetadata


def save_tif(band: np.ndarray, path: Path, meta: dict[str, Any]) -> Path:
    m = meta.copy()
    m.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)
    with rasterio.open(path, "w", **m) as dst:
        dst.write_band(1, band.astype(rasterio.uint8))
    return path


def save_png(draw_fn, path: Path, rgb: np.ndarray, title: str, background: Optional[np.ndarray] = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
    if background is not None:
        ax.imshow(background, cmap="gray")
    else:
        ax.imshow(rgb)
    draw_fn(ax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def infer_shape(
    mask: np.ndarray,
    meta: ImageMetadata,
    rgb: np.ndarray,
    out_dir: Union[Path, str],
    stem: str,
) -> dict[str, Optional[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "tif": None,
        "png": None,
        "gpkg": None,
    }

    meta_dict = {
        "dtype": rasterio.uint8,
        "count": 1,
        "height": meta.height,
        "width": meta.width,
        "transform": meta.transform,
        "crs": meta.crs,
    }

    tif_path = out_dir / f"{stem}_river_shape.tif"
    save_tif(mask, tif_path, meta_dict)
    result["tif"] = tif_path
    logger.info(f"  TIF  -> {tif_path}")

    png_path = out_dir / f"{stem}_river_shape.png"

    def draw_fn(ax):
        ov = np.zeros((meta.height, meta.width, 4), dtype=np.float32)
        ov[mask > 0] = [0.05, 0.2, 0.9, 0.55]
        ax.imshow(ov)

    save_png(draw_fn, png_path, rgb, stem)
    result["png"] = png_path
    logger.info(f"  PNG  -> {png_path}")

    bw_tif_path = out_dir / f"{stem}_river_shape_bw.tif"
    bw_mask = (mask * 255).astype(np.uint8)
    save_tif(bw_mask, bw_tif_path, meta_dict)
    logger.info(f"  BW TIF  -> {bw_tif_path}")

    bw_png_path = out_dir / f"{stem}_river_shape_bw.png"

    def draw_bw_fn(ax):
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)

    save_png(draw_bw_fn, bw_png_path, mask, stem, mask)
    logger.info(f"  BW PNG  -> {bw_png_path}")

    try:
        import geopandas as gpd
        from shapely.geometry import shape
        from shapely.ops import unary_union

        polys = [shape(g) for g, v in rio_shapes(mask, transform=meta.transform) if v == 1]
        if polys:
            gpkg_path = out_dir / f"{stem}_river_shape.gpkg"
            gpd.GeoDataFrame({"geometry": [unary_union(polys)]}, crs=meta.crs).to_file(gpkg_path, driver="GPKG")
            result["gpkg"] = gpkg_path
            logger.info(f"  GPKG -> {gpkg_path}")
    except ImportError:
        pass

    return result
