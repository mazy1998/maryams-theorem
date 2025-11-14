# Maryam's Theorem: Generalized Cauchy Projection Formula

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
pip install -r requirements.txt
```

### Compute Convexity Measure for a Single Mesh
```bash
python moeini_convexity_measure.py --mesh your_mesh.obj --samples 2000 --rays 8192
```

### Batch Process Multiple Meshes
```bash
python run_convexity_batch.py --mesh-dir toruses/elongation_1 --output results.csv
```

### Generate Test Toruses
```bash
python generate_toruses.py --elongations 1.0 2.0 3.0 4.0
```

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

## Citation

This work is dedicated to Maryam Mirzakhani and named in honor of the author's mother.
