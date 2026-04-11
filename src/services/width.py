import csv
from pathlib import Path
from typing import Any, Union, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..domain.inference import ImageMetadata

EPS = 1e-7


def measure_width_px(nc: float, nr: float, tang: np.ndarray, mask: np.ndarray, H: int, W: int) -> int:
    perp = np.array([-tang[1], tang[0]])
    perp /= np.linalg.norm(perp) + EPS
    count = 1 if (0 <= int(nr) < H and 0 <= int(nc) < W and mask[int(nr), int(nc)]) else 0
    for sign in [1, -1]:
        for s in range(1, 600):
            c = int(round(nc + sign * s * perp[0]))
            r = int(round(nr + sign * s * perp[1]))
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                count += 1
            else:
                break
    return count


def infer_width(
    mask: np.ndarray,
    skeleton: np.ndarray,
    meta: ImageMetadata,
    rgb: np.ndarray,
    out_dir: Union[Path, str],
    stem: str,
    sword_dir: Optional[Path] = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "csv": None,
        "png": None,
        "stats": None,
    }

    H, W = meta.height, meta.width
    res_m = meta.resolution
    rows = []
    sword_ok = False

    if sword_dir:
        try:
            import geopandas as gpd
            from rasterio.warp import transform_bounds
            from shapely.geometry import box as sgbox

            bwgs = transform_bounds(meta.crs, "EPSG:4326", *meta.bounds)
            bbox = sgbox(*bwgs)

            def _load(kw: str):
                hits = [f for f in Path(sword_dir).rglob("*.shp") if kw in f.name.lower()]
                if not hits:
                    return gpd.GeoDataFrame()
                g = gpd.read_file(hits[0], bbox=bwgs)
                if g.crs and g.crs.to_epsg() != 4326:
                    g = g.to_crs(epsg=4326)
                return g[g.geometry.intersects(bbox)].copy()

            reaches = _load("reach")
            nodes = _load("node")
            logger.info(f"  SWORD: {len(reaches)} reaches, {len(nodes)} nodes")

            if len(reaches) > 0 and len(nodes) > 0:
                sword_ok = True
                from shapely.geometry import LineString

                def _px(lon: float, lat: float):
                    return (lon - meta.transform.c) / meta.transform.a, (lat - meta.transform.f) / meta.transform.e

                def _tang(pt):

                    best = np.array([0.0, 1.0])
                    dmin = float("inf")
                    for _, row in reaches.iterrows():
                        g = row.geometry
                        if g is None or g.is_empty:
                            continue
                        d = g.distance(pt)
                        if d >= dmin:
                            continue
                        dmin = d
                        coords = (
                            list(g.coords)
                            if g.geom_type == "LineString"
                            else [c for seg in g.geoms for c in seg.coords]
                        )
                        bsd, bseg = float("inf"), None
                        for i in range(len(coords) - 1):
                            sd = LineString([coords[i], coords[i + 1]]).distance(pt)
                            if sd < bsd:
                                bsd, bseg = sd, (coords[i], coords[i + 1])
                        if bseg:
                            c1, c2 = _px(*bseg[0]), _px(*bseg[1])
                            dc = c2[0] - c1[0]
                            dr = c2[1] - c1[1]
                            n_ = max(np.hypot(dc, dr), EPS)
                            best = np.array([dc / n_, dr / n_])
                    return best

                for _, n in tqdm(nodes.iterrows(), total=len(nodes), desc="  SWORD"):
                    g = n.geometry
                    if g is None:
                        continue
                    nc, nr = _px(g.x, g.y)
                    if not (0 <= nc < W and 0 <= nr < H):
                        continue
                    tang = _tang(g)
                    wpx = measure_width_px(nc, nr, tang, mask, H, W)
                    rows.append(
                        {
                            "source": "SWORD",
                            "node_id": str(n.get("node_id", n.get("NODE_ID", ""))),
                            "reach_id": str(n.get("reach_id", n.get("REACH_ID", ""))),
                            "lon": round(g.x, 6),
                            "lat": round(g.y, 6),
                            "width_m": round(wpx * res_m, 1),
                            "width_px": wpx,
                            "sword_ref_m": round(float(n.get("width", 0) or 0), 1),
                            "pixel_col": int(round(nc)),
                            "pixel_row": int(round(nr)),
                        }
                    )
        except Exception as e:
            logger.warning(f"  SWORD failed ({e}) - falling back to skeleton")

    skel_pts = None
    if not sword_ok:
        logger.info("  Measuring width along skeleton (no SWORD needed)")
        skel_pts = np.argwhere(skeleton)
        n_pts = len(skel_pts)
        step = max(1, n_pts // 80)

        for i in tqdm(range(0, n_pts, step), desc="  skeleton"):
            r_i, c_i = int(skel_pts[i][0]), int(skel_pts[i][1])
            lo_ = max(0, i - 4)
            hi_ = min(n_pts - 1, i + 4)
            if hi_ > lo_:
                dc = float(skel_pts[hi_][1] - skel_pts[lo_][1])
                dr = float(skel_pts[hi_][0] - skel_pts[lo_][0])
                n_ = max(np.hypot(dc, dr), EPS)
                tang = np.array([dc / n_, dr / n_])
            else:
                tang = np.array([0.0, 1.0])
            wpx = measure_width_px(c_i, r_i, tang, mask, H, W)
            lon = meta.transform.c + (c_i + 0.5) * meta.transform.a
            lat = meta.transform.f + (r_i + 0.5) * meta.transform.e
            rows.append(
                {
                    "source": "skeleton",
                    "node_id": f"pt_{i:05d}",
                    "reach_id": "",
                    "lon": round(lon, 6),
                    "lat": round(lat, 6),
                    "width_m": round(wpx * res_m, 1),
                    "width_px": wpx,
                    "sword_ref_m": "",
                    "pixel_col": c_i,
                    "pixel_row": r_i,
                }
            )

    if rows:
        csv_path = out_dir / f"{stem}_width_numbers.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        result["csv"] = csv_path

        widths = [r["width_m"] for r in rows if r["width_m"] > 0]
        src_label = "SWORD" if sword_ok else "skeleton"
        stats = {
            "median": float(np.median(widths)),
            "mean": float(np.mean(widths)),
            "min": float(np.min(widths)),
            "max": float(np.max(widths)),
            "points": len(widths),
            "source": src_label,
        }
        result["stats"] = stats
        logger.info(f"  Median: {stats['median']:.1f}m, Mean: {stats['mean']:.1f}m")

        print("\n" + "=" * 54)
        print(f"  WIDTH RESULTS  ({len(widths)} pts via {src_label})")
        print(f"  Median : {stats['median']:.1f} m")
        print(f"  Mean   : {stats['mean']:.1f} m")
        print(f"  Min    : {stats['min']:.1f} m")
        print(f"  Max    : {stats['max']:.1f} m")
        print("=" * 54)
        print(f"\n{'node_id':<12} {'lon':>10} {'lat':>9} {'width_m':>9}")
        print("-" * 44)
        for r in rows[:30]:
            print(f"{r['node_id']:<12} {r['lon']:>10} {r['lat']:>9} {r['width_m']:>9.1f}")
        if len(rows) > 30:
            print(f"  ... {len(rows) - 30} more rows in CSV")
        print(f"\n  -> {csv_path}")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        wpng = out_dir / f"{stem}_width_numbers.png"
        fig, ax = plt.subplots(figsize=(9, 9), dpi=180)
        ax.imshow(rgb)
        ov = np.zeros((H, W, 4), dtype=np.float32)
        ov[mask > 0] = [0.05, 0.2, 0.9, 0.4]
        ax.imshow(ov)

        if skeleton is not None:
            pts = np.argwhere(skeleton)
            if len(pts):
                ax.scatter(
                    pts[:, 1],
                    pts[:, 0],
                    c="orange",
                    s=0.3,
                    linewidths=0,
                    zorder=4,
                )

        for r in rows:
            nc, nr = r["pixel_col"], r["pixel_row"]
            ax.plot(nc, nr, "o", color="yellow", ms=2, zorder=5)
            if r["width_px"] > 0 and skel_pts is not None:
                try:
                    i_s = int(r["node_id"].replace("pt_", ""))
                    lo_ = max(0, i_s - 4)
                    hi_ = min(len(skel_pts) - 1, i_s + 4)
                    dc = float(skel_pts[hi_][1] - skel_pts[lo_][1])
                    dr = float(skel_pts[hi_][0] - skel_pts[lo_][0])
                    n_ = max(np.hypot(dc, dr), EPS)
                    tang = np.array([dc / n_, dr / n_])
                except Exception:
                    tang = np.array([0.0, 1.0])
                perp = np.array([-tang[1], tang[0]])
                perp /= np.linalg.norm(perp) + EPS
                h = r["width_px"] / 2
                ax.plot(
                    [nc - h * perp[0], nc + h * perp[0]],
                    [nr - h * perp[1], nr + h * perp[1]],
                    color="yellow",
                    lw=0.8,
                    ls="--",
                    zorder=5,
                )
                ax.text(
                    nc + h * perp[0] + 3,
                    nr + h * perp[1],
                    f"{r['width_m']:.0f}m",
                    color="yellow",
                    fontsize=3.5,
                    fontweight="bold",
                    zorder=6,
                )

        ax.legend(
            handles=[
                mpatches.Patch(color=(0.05, 0.2, 0.9, 0.4), label="Water mask"),
                Line2D([0], [0], color="orange", lw=1, label="Centerline"),
                Line2D([0], [0], color="yellow", lw=1, ls="--", label="Width transect"),
            ],
            fontsize=6,
            loc="lower right",
            framealpha=0.7,
        )
        ax.set_title(f"{stem} - width numbers", fontsize=8)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(wpng, dpi=180, bbox_inches="tight")
        plt.close()
        result["png"] = wpng
        logger.info(f"  PNG  -> {wpng}")

    else:
        logger.warning("No widths produced - mask may be empty")

    return result
