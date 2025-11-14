# Maryam's Theorem: Generalized Cauchy's Surface Area Formula

This repository contains the implementation and validation of **Maryam's Theorem**, which generalizes Cauchy's projection formula to non-convex surfaces by introducing the **Moeini Convexity Measure**.

## Theorem

For any closed surface with area S, the average orthographic shadow area over all viewing directions is:

```
Ā = (S/4) × (1 - ⟨AO⟩) = (S × C_M)/4
```

where:
- `⟨AO⟩` is the area-weighted mean cosine-weighted ambient occlusion
- `C_M = 1 - ⟨AO⟩` is the Moeini Convexity Measure
- For convex shapes, `⟨AO⟩ = 0`, reducing to Cauchy's formula `Ā = S/4`

## Files

- **`Maryams_theorem.pdf`** - Complete mathematical derivation and proof
- **`moeini_convexity_measure.py`** - Core implementation for computing C_M
- **`run_convexity_batch.py`** - Batch processing script for multiple meshes
- **`generate_toruses.py`** - Generate test torus meshes with varying parameters
- **`requirements.txt`** - Python dependencies
- **`toruses/`** - Generated test meshes organized by elongation factor

## Usage

### Install Dependencies
```bash
uv pip install -r requirements.txt
```

### Compute Convexity Measure for a Single Mesh
```bash
python moeini_convexity_measure.py --mesh your_mesh.obj --samples 1000 --rays 1024 --rotations 400 
```

### Batch Process Multiple Meshes
```bash
 python run_convexity_batch.py --mesh-dir toruses/elongation_4 --samples 1000 --rays 1024 --rotations 400 --output elongated_torus_convexity_results_elon4.csv --overwrite
```

### Generate Test Toruses
```bash
python generate_toruses.py --elongations 1.0 2.0 3.0 4.0
```

## Torus Benchmark Results

The following tables summarize the measured metrics for each generated torus at
different elongation factors. Values are regenerated automatically via
`update_readme_tables.py`.

<!-- BEGIN TORUS RESULTS -->
### Elongation 1×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | MC/(S·C_M/4) | MC/(S/4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 35.87 | 0.05 | 0.95 | 8.52 | 8.59 | 1.01 | 0.96 |
| 1.25 | 31.56 | 0.07 | 0.93 | 7.37 | 7.55 | 1.02 | 0.96 |
| 1.5 | 26.30 | 0.07 | 0.93 | 6.10 | 6.10 | 1.00 | 0.93 |
| 2 | 19.73 | 0.07 | 0.93 | 4.59 | 4.68 | 1.02 | 0.95 |
| 3 | 13.15 | 0.07 | 0.93 | 3.06 | 3.19 | 1.04 | 0.97 |
| 4 | 9.86 | 0.05 | 0.95 | 2.33 | 2.35 | 1.01 | 0.95 |
| 6 | 6.58 | 0.05 | 0.95 | 1.56 | 1.62 | 1.04 | 0.99 |

### Elongation 2×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | MC/(S·C_M/4) | MC/(S/4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 55.31 | 0.08 | 0.92 | 12.79 | 12.97 | 1.01 | 0.94 |
| 1.25 | 48.67 | 0.10 | 0.90 | 10.93 | 11.15 | 1.02 | 0.92 |
| 1.5 | 40.56 | 0.12 | 0.88 | 8.93 | 8.99 | 1.01 | 0.89 |
| 2 | 30.42 | 0.13 | 0.87 | 6.59 | 6.69 | 1.01 | 0.88 |
| 3 | 20.28 | 0.14 | 0.86 | 4.38 | 4.50 | 1.03 | 0.89 |
| 4 | 15.21 | 0.12 | 0.88 | 3.35 | 3.37 | 1.01 | 0.89 |
| 6 | 10.14 | 0.10 | 0.90 | 2.28 | 2.35 | 1.03 | 0.93 |

### Elongation 3×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | MC/(S·C_M/4) | MC/(S/4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 76.30 | 0.09 | 0.91 | 17.36 | 17.66 | 1.02 | 0.93 |
| 1.25 | 67.14 | 0.12 | 0.88 | 14.74 | 15.01 | 1.02 | 0.89 |
| 1.5 | 55.95 | 0.14 | 0.86 | 11.96 | 12.05 | 1.01 | 0.86 |
| 2 | 41.96 | 0.17 | 0.83 | 8.68 | 8.76 | 1.01 | 0.83 |
| 3 | 27.98 | 0.18 | 0.82 | 5.72 | 5.82 | 1.02 | 0.83 |
| 4 | 20.98 | 0.17 | 0.83 | 4.35 | 4.40 | 1.01 | 0.84 |
| 6 | 13.99 | 0.14 | 0.86 | 3.00 | 3.05 | 1.02 | 0.87 |

### Elongation 4×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | MC/(S·C_M/4) | MC/(S/4) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 97.95 | 0.10 | 0.90 | 22.01 | 22.15 | 1.01 | 0.90 |
| 1.25 | 86.19 | 0.13 | 0.87 | 18.69 | 18.86 | 1.01 | 0.88 |
| 1.5 | 71.83 | 0.16 | 0.84 | 15.00 | 15.07 | 1.00 | 0.84 |
| 2 | 53.87 | 0.20 | 0.80 | 10.75 | 10.73 | 1.00 | 0.80 |
| 3 | 35.91 | 0.21 | 0.79 | 7.08 | 7.03 | 0.99 | 0.78 |
| 4 | 26.94 | 0.20 | 0.80 | 5.38 | 5.47 | 1.02 | 0.81 |
| 6 | 17.96 | 0.18 | 0.82 | 3.70 | 3.71 | 1.00 | 0.83 |
<!-- END TORUS RESULTS -->

## Applications

- **Thermal radiation modeling** - Effective radiative area for heat transfer
- **Computer graphics** - Ambient occlusion validation and LOD optimization  
- **Manufacturing** - Coverage prediction for coating/deposition processes
- **Robotics** - Viewpoint planning and 3D reconstruction
- **Aerospace** - Cross-sectional area for drag/ablation in tumbling objects

## Theory

The theorem provides a rigorous connection between:
1. **Geometric complexity** (surface area S)
2. **Self-occlusion** (ambient occlusion ⟨AO⟩) 
3. **Projected visibility** (average shadow area Ā)

This enables quantitative analysis of how surface concavity affects line-of-sight processes across multiple domains.


This work is dedicated to mother Maryam and to the late Maryam Mirzakhani.
