"""Monte Carlo occlusion on the Stanford bunny with a PyTorch (MPS/CUDA) ray tracer.

This script replaces trimesh's CPU ray engine with a lightweight GPU pipeline:

- Builds a simple leaf BVH (bounding boxes over small triangle batches).
- Launches hemisphere rays per surface sample, orienting them with respect to
  surface normals and testing them directly on the GPU using Möller–Trumbore.
- Interpolates per-sample results across faces and shades the mesh surface.

The goal is to exercise Apple M-series GPUs (via torch MPS) or NVIDIA GPUs
(via torch CUDA). If no GPU backend is available, the script falls back to the
CPU, but throughput will be limited.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
import trimesh

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Configuration

DEFAULT_BUNNY_PATH = Path(__file__).resolve().parent / "bunny.obj"
DEFAULT_SURFACE_SAMPLES = 1000
DEFAULT_RAYS_PER_POINT = 4096
RAY_OFFSET = 1e-3
BVH_LEAF_SIZE = 4
FACE_INTERP_K = 6
EPSILON = 1e-6


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = detect_device()


# ---------------------------------------------------------------------------
# Geometry utilities


def sample_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.clip(norms, 1e-12, None)
    return vecs.astype(np.float32)


def set_axes_equal(ax, vertices: np.ndarray) -> None:
    verts = np.asarray(vertices)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    spans = maxs - mins
    max_range = spans.max()
    if max_range <= 0:
        return
    centers = 0.5 * (mins + maxs)
    half = 0.5 * max_range
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


# ---------------------------------------------------------------------------
# BVH construction (CPU)


@dataclass
class BVH:
    bounds_min: np.ndarray  # (num_leaves, 3)
    bounds_max: np.ndarray  # (num_leaves, 3)
    leaf_tri_indices: np.ndarray  # concatenated triangle indices
    leaf_offsets: List[Tuple[int, int]]  # (start, count) per leaf


def build_leaf_bvh(triangles: np.ndarray, leaf_size: int = BVH_LEAF_SIZE) -> BVH:
    tri_indices = np.arange(len(triangles))
    centroids = triangles.mean(axis=1)

    bounds_min_list: List[np.ndarray] = []
    bounds_max_list: List[np.ndarray] = []
    offsets: List[Tuple[int, int]] = []
    tri_order: List[int] = []

    def recurse(indices: np.ndarray) -> None:
        bounds_min = triangles[indices].reshape(-1, 3).min(axis=0)
        bounds_max = triangles[indices].reshape(-1, 3).max(axis=0)
        if len(indices) <= leaf_size:
            start = len(tri_order)
            tri_order.extend(indices.tolist())
            offsets.append((start, len(indices)))
            bounds_min_list.append(bounds_min)
            bounds_max_list.append(bounds_max)
            return
        extent = bounds_max - bounds_min
        axis = int(np.argmax(extent))
        sorted_idx = np.argsort(centroids[indices, axis])
        mid = len(indices) // 2
        recurse(indices[sorted_idx[:mid]])
        recurse(indices[sorted_idx[mid:]])

    recurse(tri_indices)

    return BVH(
        bounds_min=np.asarray(bounds_min_list, dtype=np.float32),
        bounds_max=np.asarray(bounds_max_list, dtype=np.float32),
        leaf_tri_indices=np.asarray(tri_order, dtype=np.int32),
        leaf_offsets=offsets,
    )


# ---------------------------------------------------------------------------
# GPU kernels (PyTorch)


def orient_rays(base_rays: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-12)
    dots = base_rays @ normal
    dirs = torch.where(dots[:, None] >= 0, base_rays, -base_rays)
    return dirs


def orient_rays_batch(base_rays: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    normals = normals / torch.clamp(torch.linalg.norm(normals, dim=1, keepdim=True), min=1e-12)
    base = base_rays.unsqueeze(0).expand(normals.shape[0], -1, -1)
    dots = torch.sum(base * normals[:, None, :], dim=2)
    dirs = torch.where(dots.unsqueeze(-1) >= 0, base, -base)
    return dirs


def intersect_aabbs(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    inv_dirs: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
) -> torch.Tensor:
    o = origins.unsqueeze(0)
    inv = inv_dirs.unsqueeze(0)
    t1 = (bounds_min[:, None, :] - o) * inv
    t2 = (bounds_max[:, None, :] - o) * inv
    tmin = torch.minimum(t1, t2).amax(dim=2)
    tmax = torch.maximum(t1, t2).amin(dim=2)
    zeros = torch.zeros_like(tmin)
    return (tmax >= torch.maximum(tmin, zeros)) & (tmax > 0.0)


def triangle_intersections(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    v0: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
) -> torch.Tensor:
    v0 = v0.unsqueeze(0)
    e1 = e1.unsqueeze(0)
    e2 = e2.unsqueeze(0)

    pvec = torch.cross(dirs, e2.expand_as(dirs), dim=1)
    det = torch.sum(e1.expand_as(pvec) * pvec, dim=1)
    mask = torch.abs(det) > EPSILON
    inv_det = torch.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]

    tvec = origins - v0
    u = torch.sum(tvec * pvec, dim=1) * inv_det
    cond = mask & (u >= 0.0) & (u <= 1.0)
    if not torch.any(cond):
        return cond

    qvec = torch.cross(tvec, e1.expand_as(tvec), dim=1)
    v = torch.sum(dirs * qvec, dim=1) * inv_det
    cond = cond & (v >= 0.0) & ((u + v) <= 1.0)
    if not torch.any(cond):
        return cond

    t = torch.sum(e2.expand_as(qvec) * qvec, dim=1) * inv_det
    cond = cond & (t > RAY_OFFSET)
    return cond


@torch.no_grad()
def trace_rays_bvh(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    bvh: BVH,
    leaves_min: torch.Tensor,
    leaves_max: torch.Tensor,
    leaf_tris: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_e1: torch.Tensor,
    tri_e2: torch.Tensor,
) -> torch.Tensor:
    dirs_safe = torch.where(torch.abs(dirs) < 1e-8, torch.sign(dirs) * 1e-8 + 1e-8, dirs)
    inv_dirs = 1.0 / dirs_safe
    leaf_hits = intersect_aabbs(origins, dirs_safe, inv_dirs, leaves_min, leaves_max)

    hit_mask = torch.zeros(dirs.shape[0], dtype=torch.bool, device=dirs.device)

    for leaf_idx, (start, count) in enumerate(bvh.leaf_offsets):
        mask = leaf_hits[leaf_idx] & (~hit_mask)
        if not torch.any(mask):
            continue
        ray_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if ray_ids.numel() == 0:
            continue

        ray_origins = origins[ray_ids]
        ray_dirs = dirs_safe[ray_ids]

        tri_indices = leaf_tris[start : start + count]
        active_ids = ray_ids
        active_origins = ray_origins
        active_dirs = ray_dirs

        for tri_idx in tri_indices.tolist():
            if active_ids.numel() == 0:
                break
            hits = triangle_intersections(
                active_origins,
                active_dirs,
                tri_v0[tri_idx],
                tri_e1[tri_idx],
                tri_e2[tri_idx],
            )
            if torch.any(hits):
                hit_ids = active_ids[hits]
                hit_mask[hit_ids] = True
                keep = ~hits
                active_ids = active_ids[keep]
                active_origins = active_origins[keep]
                active_dirs = active_dirs[keep]

    return hit_mask


# ---------------------------------------------------------------------------
# Probability estimation pipeline


def estimate_probabilities_gpu(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    normals: np.ndarray,
    base_rays: torch.Tensor,
    bvh: BVH,
    triangles_t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    leaves_min = torch.tensor(bvh.bounds_min, device=DEVICE)
    leaves_max = torch.tensor(bvh.bounds_max, device=DEVICE)
    leaf_tris = torch.tensor(bvh.leaf_tri_indices, device=DEVICE, dtype=torch.long)
    tri_v0, tri_e1, tri_e2 = triangles_t

    probabilities = np.empty(points.shape[0], dtype=np.float64)

    num_points = points.shape[0]
    rays_count = base_rays.shape[0]

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch_points = points[start:end]
        batch_normals = normals[start:end]
        batch_len = end - start

        normals_t = torch.tensor(batch_normals, device=DEVICE, dtype=torch.float32)
        dirs = orient_rays_batch(base_rays, normals_t)
        origins_np = batch_points + batch_normals * RAY_OFFSET
        origins_t = torch.tensor(origins_np, device=DEVICE, dtype=torch.float32).unsqueeze(1).expand(-1, rays_count, -1)

        flat_dirs = dirs.reshape(-1, 3)
        flat_origins = origins_t.reshape(-1, 3)

        hit_mask = trace_rays_bvh(
            flat_origins,
            flat_dirs,
            bvh,
            leaves_min,
            leaves_max,
            leaf_tris,
            tri_v0,
            tri_e1,
            tri_e2,
        )

        hit_mask = hit_mask.view(batch_len, rays_count)
        batch_probs = 0.5 + 0.5 * hit_mask.float().mean(dim=1).cpu().numpy()
        probabilities[start:end] = batch_probs

        processed = end
        if processed % max(1, num_points // 10) == 0 or processed == num_points:
            print(f"  processed {processed}/{num_points} surface samples")

    return probabilities


# ---------------------------------------------------------------------------
# CLI and plotting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU Monte Carlo occlusion on Stanford bunny")
    parser.add_argument("--bunny", type=Path, default=DEFAULT_BUNNY_PATH, help="Path to bunny OBJ")
    parser.add_argument("--samples", type=int, default=DEFAULT_SURFACE_SAMPLES, help="Surface samples")
    parser.add_argument("--rays", type=int, default=DEFAULT_RAYS_PER_POINT, help="Rays per sample")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--leaf-size", type=int, default=BVH_LEAF_SIZE, help="Triangles per BVH leaf")
    parser.add_argument("--batch-size", type=int, default=16, help="Surface samples processed together")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.bunny.exists():
        raise FileNotFoundError(f"Bunny mesh not found at {args.bunny}")

    print(f"Using torch device: {DEVICE}")
    print(f"Loading mesh from {args.bunny} ...")
    mesh = trimesh.load_mesh(args.bunny, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.as_trimesh()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.rezero()
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    print("Sampling surface points ...")
    points, face_ids = trimesh.sample.sample_surface_even(mesh, args.samples)
    normals = mesh.face_normals[face_ids]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print("Preparing BVH ...")
    triangles = mesh.triangles.astype(np.float32)
    bvh = build_leaf_bvh(triangles, leaf_size=args.leaf_size)
    tri_ordered = triangles[bvh.leaf_tri_indices]
    tri_v0 = torch.tensor(tri_ordered[:, 0, :], device=DEVICE)
    tri_e1 = torch.tensor(tri_ordered[:, 1, :] - tri_ordered[:, 0, :], device=DEVICE)
    tri_e2 = torch.tensor(tri_ordered[:, 2, :] - tri_ordered[:, 0, :], device=DEVICE)

    print("Generating base rays ...")
    base_rays_np = sample_unit_vectors(args.rays, rng)
    base_rays = torch.tensor(base_rays_np, device=DEVICE)

    print("Tracing rays on GPU ...")
    start = time.perf_counter()
    probabilities = estimate_probabilities_gpu(
        mesh,
        points,
        normals,
        base_rays,
        bvh,
        (tri_v0, tri_e1, tri_e2),
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - start
    print(f"Monte Carlo sweep completed in {elapsed:.2f}s")

    print("Interpolating face probabilities ...")
    face_centers = mesh.triangles_center
    k = min(FACE_INTERP_K, max(1, len(points)))
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

    print("=== GPU Bunny Occlusion Statistics ===")
    print(f"Minimum probability: {min_prob:.4f}")
    print(f"Maximum probability: {max_prob:.4f}")
    print(f"Mean probability: {mean_prob:.4f}")
    print(f"Standard deviation: {std_prob:.4f}")
    print(f"Median probability: {median_prob:.4f}")
    print(f"Range: {max_prob - min_prob:.4f}")
    print(f"Total surface samples: {len(probabilities)}")
    print(f"Total mesh faces shaded: {len(face_probs)}")

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    norm = Normalize(vmin=min_prob, vmax=max_prob)
    face_colors = plt.cm.viridis(norm(face_probs))
    collection = Poly3DCollection(mesh.triangles, facecolors=face_colors, linewidths=0.05, edgecolors="none")
    ax1.add_collection3d(collection)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    mappable.set_array(face_probs)
    fig.colorbar(mappable, ax=ax1, shrink=0.6, aspect=12, label="Probability")
    ax1.set_title("GPU Monte Carlo Bunny")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    set_axes_equal(ax1, mesh.vertices)

    ax2 = fig.add_subplot(122)
    n_bins = min(50, max(12, len(stats_values) // 25))
    counts, bins, patches = ax2.hist(stats_values, bins=n_bins, alpha=0.75, edgecolor="black", color="skyblue")
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

