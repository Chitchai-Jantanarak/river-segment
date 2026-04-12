# River Segment

River width inference from satellite imagery using deep learning.

## Installation

```bash
uv sync
```

## Quick Start

### Inference Scripts

```bash
# Shape inference (water mask)
python infer-shape.py --input data/foo.tif --ckpt model/baz.pth.tar

# Centerline inference
python infer-centerline.py --input data/foo.tif --ckpt model/baz.pth.tar

# Width inference
python infer-width.py --input data/foo.tif --ckpt model/baz.pth.tar

# All tasks
python infer-all.py --input data/foo.tif --ckpt model/baz.pth.tar
```

### Arguments

| Argument | Description | Default |
|-----------|-------------|---------|
| `--input` | Input GeoTIFF path | Required |
| `--ckpt` | Model checkpoint path | Required |
| `--out` | Output directory | `results` |
| `--thresh` | Water probability threshold (0-1) | 0.4 |
| `--size` | Model input size | 512 |

### Example Cases

#### 1. Basic Shape Inference

```bash
python infer-shape.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar
```

Output:
- `results/foo_river_shape.tif`
- `results/foo_river_shape.png`
- `results/foo_river_shape.gpkg`
- `results/foo_river_shape_bw.tif` (binary mask)
- `results/foo_river_shape_bw.png` (grayscale)

#### 2. Lower Threshold (more water detected)

```bash
python infer-shape.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar \
    --thresh 0.2
```

#### 3. Custom Output Directory

```bash
python infer-shape.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar \
    --out output/river1
```

#### 4. Centerline Only

```bash
python infer-centerline.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar \
    --out results
```

#### 5. Width Measurements

```bash
python infer-width.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar \
    --thresh 0.4
```

#### 6. All Tasks at Once

```bash
python infer-all.py \
    --input data/foo.tif \
    --ckpt model/baz.pth.tar \
    --thresh 0.4 \
    --out results
```

### Training

```bash
python finetune.py \
    --data ./data \
    --ckpt model/baz.pth.tar \
    --out model
```

Training arguments:
- `--data` - Dataset directory
- `--ckpt` - Starting checkpoint
- `--out` - Output directory
- `--epochs` - Number of epochs (default: 30)
- `--lr` - Learning rate (default: 1e-4)
- `--batch` - Batch size (default: 8)
- `--warmup` - Warmup epochs (default: 5)

## Output Files

Each inference produces:

- `{name}_river_shape.tif` - Water mask (RGB)
- `{name}_river_shape.png` - Water mask overlay
- `{name}_river_shape_bw.tif` - Binary mask (0/1)
- `{name}_river_shape_bw.png` - Binary mask (grayscale)
- `{name}_river_shape.gpkg` - GeoPackage vector

Centerline additionally:
- `{name}_river_centerline.tif` - Skeleton centerline
- `{name}_river_centerline.png` - Centerline overlay

Width additionally:
- `{name}_width_numbers.csv` - Width measurements
- `{name}_width_numbers.png` - Width visualization

## Project Structure

```
river-segment/
├── config/hydra/          # Hydra configs
├── src/river_segment/     # Python package
│   ├── domain/            # DTOs
│   ├── data/               # Data loading
│   ├── services/          # Inference services
│   └── models/             # Model factory
├── data/                  # Input data
├── model/                  # Checkpoints
├── results/                # Output results
├── infer-shape.py          # Shape inference
├── infer-centerline.py     # Centerline inference
├── infer-width.py          # Width inference
├── infer-all.py             # All tasks
└── finetune.py            # Training
```
