"""Monte Carlo occlusion probabilities on the Stanford bunny mesh.

This script mirrors the hemisphere-based Monte Carlo routine used for the torus,
but evaluates the probabilities over an arbitrary triangle mesh (Stanford bunny).
For each sampled point on the mesh surface we shoot a large number of random
rays distributed uniformly over the outward hemisphere and record the
probability that the rays intersect the mesh. The lower hemisphere contributes
exactly 0.5 by symmetry, so the reported probability is `0.5 + 0.5 * hit_rate`.

CLI options let you tune the number of surface samples and rays per sample.
The output includes summary statistics, a mesh rendering shaded by probability,
and a histogram of the distribution.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
import trimesh


# ---------------------------------------------------------------------------
# Configuration

DEFAULT_BUNNY_PATH = Path(__file__).resolve().parent / "Teapot.obj"
DEFAULT_SURFACE_SAMPLES = 1200
DEFAULT_RAYS_PER_POINT = 4096
SURFACE_OFFSET = 1e-3
FACE_SMOOTH_K = 5


# ---------------------------------------------------------------------------
# Utility helpers

def set_axes_equal(ax, points: np.ndarray) -> None:
    """Set equal scaling for a 3D axis given point cloud data."""

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = maxs - mins
    max_range = spans.max()

    if max_range == 0:
        return

    centers = 0.5 * (mins + maxs)
    half = 0.5 * max_range

    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def sample_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `count` unit vectors uniformly on the sphere."""

    vecs = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.clip(norms, 1e-12, None)
    return vecs.astype(np.float64)


def orient_to_hemisphere(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Flip vectors so that all lie in the hemisphere defined by `normal`."""

    normal = normal / np.linalg.norm(normal)
    dots = vectors @ normal
    signs = np.where(dots >= 0.0, 1.0, -1.0)
    return vectors * signs[:, None]


def load_bunny_mesh(path: Path) -> trimesh.Trimesh:
    """Load the Stanford bunny mesh and ensure it is a single watertight mesh."""

    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.as_trimesh()

    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.rezero()
    return mesh


def compute_hit_probability(
    ray_engine: trimesh.ray.ray_triangle.RayMeshIntersector,
    point: np.ndarray,
    normal: np.ndarray,
    base_rays: np.ndarray,
    surface_offset: float,
) -> float:
    """Return the fraction of hemisphere rays that intersect the mesh."""

    directions = orient_to_hemisphere(base_rays, normal)
    origin = point + normal / np.linalg.norm(normal) * surface_offset
    origins = np.broadcast_to(origin, directions.shape).astype(np.float64, copy=True)
    hits = ray_engine.intersects_any(origins, directions)
    return float(np.count_nonzero(hits)) / directions.shape[0]


def estimate_probabilities(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    normals: np.ndarray,
    rays: np.ndarray,
    surface_offset: float,
) -> np.ndarray:
    ray_engine = mesh.ray
    probabilities = np.empty(points.shape[0], dtype=np.float64)

    progress_mod = max(1, points.shape[0] // 10)
    start = time.perf_counter()

    for idx, (point, normal) in enumerate(zip(points, normals), start=1):
        hit_fraction = compute_hit_probability(ray_engine, point, normal, rays, surface_offset)
        probabilities[idx - 1] = 0.5 + 0.5 * hit_fraction

        if idx % progress_mod == 0:
            elapsed = time.perf_counter() - start
            print(f"  processed {idx}/{points.shape[0]} samples (elapsed {elapsed:.2f}s)")

    return probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo occlusion on Stanford bunny")
    parser.add_argument(
        "--bunny",
        type=Path,
        default=DEFAULT_BUNNY_PATH,
        help="Path to the Stanford bunny OBJ file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SURFACE_SAMPLES,
        help="Number of surface samples",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=DEFAULT_RAYS_PER_POINT,
        help="Number of hemisphere rays per surface sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.bunny.exists():
        raise FileNotFoundError(f"Bunny mesh not found at {args.bunny}")

    print(f"Loading bunny mesh from {args.bunny} ...")
    mesh = load_bunny_mesh(args.bunny)
    print(
        f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
        f"watertight={mesh.is_watertight}"
    )

    print(f"Sampling {args.samples} surface points ...")
    points, face_indices = trimesh.sample.sample_surface_even(mesh, args.samples)
    face_normals = mesh.face_normals[face_indices]
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    print(f"Generating {args.rays} random rays ...")
    base_rays = sample_unit_vectors(args.rays, rng)

    print("Estimating occlusion probabilities ...")
    start = time.perf_counter()
    probabilities = estimate_probabilities(mesh, points, face_normals, base_rays, SURFACE_OFFSET)
    elapsed = time.perf_counter() - start
    print(f"Monte Carlo sweep completed in {elapsed:.2f}s")

    print("Interpolating face probabilities ...")
    face_centers = mesh.triangles_center
    k = min(FACE_SMOOTH_K, max(1, len(points)))
    tree = cKDTree(points)
    distances, indices = tree.query(face_centers, k=k)

    if k == 1:
        face_probs = probabilities[indices]
    else:
        distances = np.clip(distances, 1e-6, None)
        weights = 1.0 / distances
        face_probs = np.sum(probabilities[indices] * weights, axis=1) / np.sum(weights, axis=1)

    stats_values = face_probs

    min_prob = float(stats_values.min())
    max_prob = float(stats_values.max())
    mean_prob = float(stats_values.mean())
    std_prob = float(stats_values.std())
    median_prob = float(np.median(stats_values))
    face_areas = mesh.area_faces
    total_area = float(face_areas.sum())
    if total_area > 0:
        weighted_mean = float(np.average(face_probs, weights=face_areas))
        weighted_std = float(
            np.sqrt(np.average((face_probs - weighted_mean) ** 2, weights=face_areas))
        )
    else:
        weighted_mean = float("nan")
        weighted_std = float("nan")

    print("=== Ray Hit Probability Statistics (Bunny) ===")
    print(f"Minimum probability: {min_prob:.4f}")
    print(f"Maximum probability: {max_prob:.4f}")
    print(f"Mean probability: {mean_prob:.4f}")
    print(f"Standard deviation: {std_prob:.4f}")
    print(f"Area-weighted mean: {weighted_mean:.4f}")
    print(f"Area-weighted std.: {weighted_std:.4f}")
    print(f"Median probability: {median_prob:.4f}")
    print(f"Range: {max_prob - min_prob:.4f}")
    print(f"Total surface samples: {len(probabilities)}")
    print(f"Total mesh faces shaded: {len(face_probs)}")

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    norm = Normalize(vmin=min_prob, vmax=max_prob)
    face_colors = plt.cm.viridis(norm(face_probs))
    collection = Poly3DCollection(
        mesh.triangles,
        facecolors=face_colors,
        linewidths=0.05,
        edgecolors="none",
        alpha=1.0,
    )
    ax1.add_collection3d(collection)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    mappable.set_array(face_probs)
    fig.colorbar(mappable, ax=ax1, shrink=0.6, aspect=12, label="Probability")
    ax1.set_title("Stanford Bunny Occlusion Probability")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    set_axes_equal(ax1, mesh.vertices)

    ax2 = fig.add_subplot(122)
    n_bins = min(40, max(10, len(stats_values) // 20))
    counts, bins, patches = ax2.hist(
        stats_values,
        bins=n_bins,
        alpha=0.75,
        edgecolor="black",
        color="skyblue",
    )

    for count, left, right, patch in zip(counts, bins[:-1], bins[1:], patches):
        center = 0.5 * (left + right)
        normalized = 0.0 if max_prob == min_prob else (center - min_prob) / (max_prob - min_prob)
        patch.set_facecolor(plt.cm.viridis(normalized))

    ax2.axvline(mean_prob, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_prob:.3f}")
    ax2.axvline(median_prob, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_prob:.3f}")
    ax2.axvline(min_prob, color="blue", linestyle=":", linewidth=1, label=f"Min: {min_prob:.3f}")
    ax2.axvline(max_prob, color="green", linestyle=":", linewidth=1, label=f"Max: {max_prob:.3f}")
    ax2.set_xlabel("Ray Hit Probability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Probability Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

