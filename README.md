# Maryam's Theorem: Generalized Cauchy's Surface Area Formula

This repository provides the PDF, code, and experimental validation for **Maryam’s Theorem**, a generalization of Cauchy’s projection formula to non-convex surfaces via the **Moeini Convexity Measure**. The proof sits on a differential-geometry foundation but leverages concepts from computer graphics and radiance fields to achieve the end result. 

[AO Shaded Bunny Rotation](visualizations/bunny/mesh_rotation.mp4)

## Theorem

For any closed surface $K \subset \mathbb{R}^3$ with total surface area $S := \int_{\partial K} \mathrm{d}A$, the average orthographic shadow area over all viewing directions is

$$
\bar{A} = \frac{S}{4}\left(1 - \langle AO \rangle\right) = \frac{S \cdot C_M}{4}.
$$

Where

$$
\langle AO \rangle = \frac{1}{S} \int_{\partial K} AO(\mathbf{x}) \ \mathrm{d}A(\mathbf{x})
$$

is the surface-area-weighted mean cosine-weighted ambient occlusion, and $C_M = 1 - \langle AO \rangle$ is the Moeini Convexity Measure. For convex shapes, $\langle AO \rangle = 0$, so the expression collapses to Cauchy’s classical result $\bar{A} = S/4$.

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

With the recommended parameters, the Monte Carlo benchmark proceeds as follows. First, 1,000 surface points are sampled, and for each point 1,024 hemispherical rays are traced to estimate the cosine-weighted ambient occlusion, yielding the data needed for $\bar{A} = \frac{S \cdot C_M}{4}$. Next, the mesh undergoes 400 random rotations; for each orientation we compute the orthographic shadow area. The resulting empirical mean shadow is then compared against Maryam’s theorem prediction, providing a direct validation of $\frac{S \cdot C_M}{4}$. 

Procedural torus meshes were used for numerical validation because raw scans often contained holes and other artifacts. To stress-test the theorem, we generated 28 toruses spanning multiple aspect ratios and elongation factors. Maryam’s theorem matched the Monte Carlo measurements almost perfectly across all cases; the small deviations are attributable to Monte Carlo variance rather than any systematic bias.



<!-- BEGIN TORUS RESULTS -->
### Elongation 1×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | Maryam's Theorem Error | Cauchy Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 35.87 | 0.05 | 0.95 | 8.52 | 8.59 | 0.01 | 0.04 |
| 1.25 | 31.56 | 0.07 | 0.93 | 7.37 | 7.55 | 0.02 | 0.04 |
| 1.5 | 26.30 | 0.07 | 0.93 | 6.10 | 6.10 | 0.00 | 0.07 |
| 2 | 19.73 | 0.07 | 0.93 | 4.59 | 4.68 | 0.02 | 0.05 |
| 3 | 13.15 | 0.07 | 0.93 | 3.06 | 3.19 | 0.04 | 0.03 |
| 4 | 9.86 | 0.05 | 0.95 | 2.33 | 2.35 | 0.01 | 0.05 |
| 6 | 6.58 | 0.05 | 0.95 | 1.56 | 1.62 | 0.04 | 0.01 |

### Elongation 2×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | Maryam's Theorem Error | Cauchy Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 55.31 | 0.08 | 0.92 | 12.79 | 12.97 | 0.01 | 0.06 |
| 1.25 | 48.67 | 0.10 | 0.90 | 10.93 | 11.15 | 0.02 | 0.08 |
| 1.5 | 40.56 | 0.12 | 0.88 | 8.93 | 8.99 | 0.01 | 0.11 |
| 2 | 30.42 | 0.13 | 0.87 | 6.59 | 6.69 | 0.01 | 0.12 |
| 3 | 20.28 | 0.14 | 0.86 | 4.38 | 4.50 | 0.03 | 0.11 |
| 4 | 15.21 | 0.12 | 0.88 | 3.35 | 3.37 | 0.01 | 0.11 |
| 6 | 10.14 | 0.10 | 0.90 | 2.28 | 2.35 | 0.03 | 0.07 |

### Elongation 3×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | Maryam's Theorem Error | Cauchy Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 76.30 | 0.09 | 0.91 | 17.36 | 17.66 | 0.02 | 0.07 |
| 1.25 | 67.14 | 0.12 | 0.88 | 14.74 | 15.01 | 0.02 | 0.11 |
| 1.5 | 55.95 | 0.14 | 0.86 | 11.96 | 12.05 | 0.01 | 0.14 |
| 2 | 41.96 | 0.17 | 0.83 | 8.68 | 8.76 | 0.01 | 0.17 |
| 3 | 27.98 | 0.18 | 0.82 | 5.72 | 5.82 | 0.02 | 0.17 |
| 4 | 20.98 | 0.17 | 0.83 | 4.35 | 4.40 | 0.01 | 0.16 |
| 6 | 13.99 | 0.14 | 0.86 | 3.00 | 3.05 | 0.02 | 0.13 |

### Elongation 4×

| Aspect Ratio | Surface Area | ⟨AO⟩ | C_M | S·C_M/4 | MC Shadow | Maryam's Theorem Error | Cauchy Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 97.95 | 0.10 | 0.90 | 22.01 | 22.15 | 0.01 | 0.10 |
| 1.25 | 86.19 | 0.13 | 0.87 | 18.69 | 18.86 | 0.01 | 0.12 |
| 1.5 | 71.83 | 0.16 | 0.84 | 15.00 | 15.07 | 0.00 | 0.16 |
| 2 | 53.87 | 0.20 | 0.80 | 10.75 | 10.73 | 0.00 | 0.20 |
| 3 | 35.91 | 0.21 | 0.79 | 7.08 | 7.03 | 0.01 | 0.22 |
| 4 | 26.94 | 0.20 | 0.80 | 5.38 | 5.47 | 0.02 | 0.19 |
| 6 | 17.96 | 0.18 | 0.82 | 3.70 | 3.71 | 0.00 | 0.17 |
<!-- END TORUS RESULTS -->

## Applications

- **Thermal radiation modeling** - Effective radiative area for heat transfer
- **Computer graphics** - Ambient occlusion validation and LOD optimization  
- **Manufacturing** - Estimating the manufacturing complexity of shapes


## Theory

The theorem provides a rigorous connection between:
1. **Geometric complexity** (surface area S)
2. **Self-occlusion** (ambient occlusion ⟨AO⟩) 
3. **Projected visibility** (average shadow area Ā)

This enables quantitative analysis of how surface concavity affects line-of-sight processes across multiple domains.


This work is dedicated to my mother, Maryam, and to the late Maryam Mirzakhani. 

Special thanks to OpenAI's Codex and GPT for their awesome coding models.
