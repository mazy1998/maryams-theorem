"""
Batch runner for Moeini Convexity Measure computations.

This script scans a directory for OBJ meshes (toruses by default), evaluates the
Moeini Convexity Measure for each mesh using the routines from
`moeini_convexity_measure.py`, and records the results in a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import moeini_convexity_measure as mcm


@dataclass(frozen=True)
class BatchConfig:
    mesh_dir: Path
    pattern: str
    output_csv: Path
    samples: int
    rays_per_sample: int
    rotations: int
    seed: int
    overwrite: bool


@dataclass
class BatchResult:
    mesh: str
    surface_area: float
    mean_ao: float
    moeini_convexity: float
    expected_shadow: float
    monte_carlo_shadow: float
    ratio_vs_expected: float
    ratio_vs_cauchy: float
    error_vs_expected: float
    error_vs_cauchy: float


def parse_args() -> BatchConfig:
    parser = argparse.ArgumentParser(
        description="Run Moeini Convexity Measure on a batch of meshes."
    )
    default_dir = Path(__file__).resolve().parent / "toruses"
    default_csv = Path(__file__).resolve().parent / "torus_convexity_results.csv"
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=default_dir,
        help="Directory containing OBJ meshes to process.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.obj",
        help="Glob pattern for selecting meshes within the directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_csv,
        help="Path to the CSV file where results are saved.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=mcm.DEFAULT_SURFACE_SAMPLES,
        help="Number of surface samples per mesh.",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=mcm.DEFAULT_RAYS_PER_SAMPLE,
        help="Number of hemisphere rays per sample.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=mcm.DEFAULT_ROTATIONS,
        help="Monte Carlo rotation count for shadow averaging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed used for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    args = parser.parse_args()

    config = BatchConfig(
        mesh_dir=args.mesh_dir,
        pattern=args.pattern,
        output_csv=args.output,
        samples=args.samples,
        rays_per_sample=args.rays,
        rotations=args.rotations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    return config


def gather_meshes(directory: Path, pattern: str) -> List[Path]:
    meshes = sorted(directory.glob(pattern))
    return meshes


def interpolate_face_occlusion(
    points: np.ndarray,
    samples: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """Interpolate occlusion estimates from sampled points to faces."""
    face_centers = mesh.triangles_center
    k = min(mcm.FACE_SMOOTH_K, max(1, len(points)))

    if len(points) == 0:
        raise ValueError("No surface samples available for interpolation.")

    tree = cKDTree(points)
    distances, indices = tree.query(face_centers, k=k)

    if k == 1:
        face_prob = samples[indices]
    else:
        distances = np.clip(distances, 1e-6, None)
        weights = 1.0 / distances
        face_prob = np.sum(samples[indices] * weights, axis=1) / np.sum(weights, axis=1)

    return face_prob


def compute_metrics(mesh_path: Path, config: BatchConfig, seed: int) -> BatchResult:
    """Compute Moeini Convexity metrics for a single mesh."""
    print(f"Loading mesh from {mesh_path} ...")
    mesh = mcm.load_mesh(mesh_path)
    area_faces = mesh.area_faces
    total_area = float(area_faces.sum())
    print(
        f"  vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
        f"surface_area={total_area:.6f}"
    )

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"  Sampling {config.samples} surface points ...")
    points, face_indices = trimesh.sample.sample_surface_even(mesh, config.samples)
    normals = mesh.face_normals[face_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print(f"  Generating {config.rays_per_sample} random rays ...")
    base_rays = mcm.sample_unit_vectors(config.rays_per_sample, rng)

    print("  Estimating occlusion probabilities ...")
    occlusion_samples = mcm.estimate_occlusion(
        mesh,
        points,
        normals,
        base_rays,
        mcm.SURFACE_OFFSET,
    )

    print("  Interpolating occlusion to faces ...")
    face_prob = interpolate_face_occlusion(points, occlusion_samples, mesh)
    face_ao = 2.0 * face_prob - 1.0

    if total_area > 0:
        mean_ao = float(np.average(face_ao, weights=area_faces))
    else:
        mean_ao = float("nan")

    moeini_convexity = 1.0 - mean_ao if math.isfinite(mean_ao) else float("nan")

    if math.isfinite(moeini_convexity) and total_area > 0:
        expected_shadow = (total_area * moeini_convexity) / 4.0
    else:
        expected_shadow = float("nan")

    print(f"  Monte Carlo averaging over {config.rotations} rotations ...")
    shadow_mean = mcm.monte_carlo_shadow(mesh, rotations=config.rotations, rng=rng)

    ratio_vs_expected = (
        shadow_mean / expected_shadow
        if expected_shadow and math.isfinite(expected_shadow) and expected_shadow != 0.0
        else float("nan")
    )

    cauchy_shadow = (total_area / 4.0) if total_area > 0 else float("nan")
    ratio_vs_cauchy = (
        shadow_mean / cauchy_shadow if cauchy_shadow and math.isfinite(cauchy_shadow) else float("nan")
    )

    error_vs_expected = (
        abs(ratio_vs_expected - 1.0) if math.isfinite(ratio_vs_expected) else float("nan")
    )
    error_vs_cauchy = (
        abs(ratio_vs_cauchy - 1.0) if math.isfinite(ratio_vs_cauchy) else float("nan")
    )

    print(f"  S/4 (Cauchy):          {cauchy_shadow:.6f}")
    print(f"  S * C_M / 4:           {expected_shadow:.6f}")
    print(f"  Monte Carlo shadow:    {shadow_mean:.6f}")
    if math.isfinite(ratio_vs_expected):
        print(f"  MC / (S*C_M/4):        {ratio_vs_expected:.6f}")
        print(f"    abs error:           {error_vs_expected:.6f}")
    if math.isfinite(ratio_vs_cauchy):
        print(f"  MC / (S/4):            {ratio_vs_cauchy:.6f}")
        print(f"    abs error:           {error_vs_cauchy:.6f}")

    result = BatchResult(
        mesh=mesh_path.name,
        surface_area=total_area,
        mean_ao=mean_ao,
        moeini_convexity=moeini_convexity,
        expected_shadow=expected_shadow,
        monte_carlo_shadow=shadow_mean,
        ratio_vs_expected=ratio_vs_expected,
        ratio_vs_cauchy=ratio_vs_cauchy,
        error_vs_expected=error_vs_expected,
        error_vs_cauchy=error_vs_cauchy,
    )
    return result


def main() -> None:
    config = parse_args()

    if not config.mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {config.mesh_dir}")

    meshes = gather_meshes(config.mesh_dir, config.pattern)
    if not meshes:
        raise FileNotFoundError(
            f"No meshes found in {config.mesh_dir} matching pattern '{config.pattern}'."
        )

    if config.output_csv.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output file {config.output_csv} exists. Pass --overwrite to replace it."
        )

    fieldnames = [
        "mesh",
        "surface_area",
        "mean_ao",
        "moeini_convexity",
        "expected_shadow",
        "monte_carlo_shadow",
        "ratio_vs_expected",
        "ratio_vs_cauchy",
        "error_vs_expected",
        "error_vs_cauchy",
    ]

    with config.output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        for idx, mesh_path in enumerate(meshes, start=1):
            print(f"[{idx}/{len(meshes)}] Processing {mesh_path.name}")
            seed = config.seed + idx
            result = compute_metrics(mesh_path, config, seed)
            writer.writerow(asdict(result))
            csvfile.flush()
            print(
                f"  mean_ao={result.mean_ao:.6f} "
                f"C_M={result.moeini_convexity:.6f} "
                f"shadow_mc={result.monte_carlo_shadow:.6f} "
                f"|err_expected|={result.error_vs_expected:.6f} "
                f"|err_cauchy|={result.error_vs_cauchy:.6f}"
            )

    print(f"Results saved to {config.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

