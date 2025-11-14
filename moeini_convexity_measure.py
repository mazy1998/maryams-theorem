"""Compute the Moeini Convexity Measure (C_M) for a triangle mesh.

C_M := 1 - <AO>, where <AO> is the surface-area-weighted mean of the
cosine-weighted hemispherical ambient occlusion as defined in the accompanying
derivation of Maryam's theorem.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh
from scipy.spatial import cKDTree


DEFAULT_MESH_PATH = Path(__file__).resolve().parent / "Teapot.obj"


def project_shadow_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Orthographic shadow area onto the XY-plane."""
    polygons = []
    for face in faces:
        # Discard the z-component so each triangle projects onto the XY-plane.
        tri = vertices[face, :2]
        area = 0.5 * abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )
        if area < 1e-12:
            # Very skinny projected triangles are numerically unstable; skip them.
            continue
        polygons.append(Polygon(tri))
    if not polygons:
        return 0.0
    # The shadow area is the area of the union of all projected triangles.
    return float(unary_union(polygons).area)
# Monte Carlo / numeric defaults. Increase these for higher precision at the cost
# of additional runtime; decrease them for faster but noisier estimates.
DEFAULT_SURFACE_SAMPLES = 1200
DEFAULT_RAYS_PER_SAMPLE = 4096
# Offset rays slightly along the outward normal to avoid self-intersections.
SURFACE_OFFSET = 1e-5
# How many nearest samples to blend when building per-face AO estimates.
FACE_SMOOTH_K = 5
DEFAULT_ROTATIONS = 1000


def sample_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw random directions uniformly on the unit sphere."""
    vecs = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.clip(norms, 1e-12, None)
    return vecs.astype(np.float64)


def orient_to_hemisphere(vectors: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Flip any backward-facing ray so it lies within the outward hemisphere."""
    normal = normal / np.linalg.norm(normal)
    dots = vectors @ normal
    signs = np.where(dots >= 0.0, 1.0, -1.0)
    return vectors * signs[:, None]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.as_trimesh()
    # Clean up common issues so sampling, normals, and ray queries are reliable.
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.rezero()
    return mesh


def compute_hit_fraction(
    ray_engine: trimesh.ray.ray_triangle.RayMeshIntersector,
    point: np.ndarray,
    normal: np.ndarray,
    base_rays: np.ndarray,
    surface_offset: float,
) -> float:
    # Reflect the shared direction set so every sample uses outward-pointing rays.
    directions = orient_to_hemisphere(base_rays, normal)
    # Offset each origin slightly along the normal to avoid self-hits.
    origin = point + normal / np.linalg.norm(normal) * surface_offset
    origins = np.broadcast_to(origin, directions.shape).astype(np.float64, copy=True)
    hits = ray_engine.intersects_any(origins, directions)
    return float(np.count_nonzero(hits)) / directions.shape[0]


def estimate_occlusion(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    normals: np.ndarray,
    base_rays: np.ndarray,
    surface_offset: float,
) -> np.ndarray:
    ray_engine = mesh.ray
    probabilities = np.empty(points.shape[0], dtype=np.float64)

    progress_mod = max(1, points.shape[0] // 10)
    start = time.perf_counter()

    for idx, (point, normal) in enumerate(zip(points, normals), start=1):
        hit_fraction = compute_hit_fraction(ray_engine, point, normal, base_rays, surface_offset)
        # Map hit fraction in [0, 1] to cosine-weighted AO within [0, 1].
        probabilities[idx - 1] = 0.5 + 0.5 * hit_fraction

        if idx % progress_mod == 0:
            elapsed = time.perf_counter() - start
            print(f"  processed {idx}/{points.shape[0]} samples (elapsed {elapsed:.2f}s)")

    return probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Moeini Convexity Measure (C_M) for a mesh."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH_PATH,
        help="Path to the mesh file (OBJ/STL/etc.).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SURFACE_SAMPLES,
        help="Number of surface samples for occlusion estimation.",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=DEFAULT_RAYS_PER_SAMPLE,
        help="Number of hemisphere rays per surface sample.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=DEFAULT_ROTATIONS,
        help="Number of random orientations for the shadow average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def monte_carlo_shadow(mesh: trimesh.Trimesh, rotations: int, rng: np.random.Generator) -> float:
    """Average shadow area via random rotations."""
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    quats = rng.normal(size=(rotations, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    shadows = np.empty(rotations, dtype=np.float64)
    start = time.perf_counter()
    for idx, quat in enumerate(quats, start=1):
        x, y, z, w = quat
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        # Convert the unit quaternion to a 3x3 rotation matrix.
        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        rotated = vertices @ R.T
        shadows[idx - 1] = project_shadow_area(rotated, faces)
        if idx % max(1, rotations // 8) == 0:
            elapsed = time.perf_counter() - start
            print(f"  shadow samples {idx}/{rotations} (elapsed {elapsed:.2f}s)")
    return float(shadows.mean())


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.mesh.exists():
        raise FileNotFoundError(f"Mesh not found at {args.mesh}")

    print(f"Loading mesh from {args.mesh} ...")
    mesh = load_mesh(args.mesh)
    vertices = mesh.vertices
    faces = mesh.faces
    area_faces = mesh.area_faces
    total_area = float(area_faces.sum())
    print(
        f"Loaded mesh with {len(vertices)} vertices, {len(faces)} faces, "
        f"surface area = {total_area:.6f}"
    )

    print(f"Sampling {args.samples} surface points ...")
    # Draw surface samples proportionally to area, returning both positions and face indices.
    points, face_indices = trimesh.sample.sample_surface_even(mesh, args.samples)
    normals = mesh.face_normals[face_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print(f"Generating {args.rays} random rays ...")
    base_rays = sample_unit_vectors(args.rays, rng)

    print("Estimating cosine-weighted occlusion probabilities ...")
    samples = estimate_occlusion(mesh, points, normals, base_rays, SURFACE_OFFSET)

    print("Interpolating per-face occlusion probabilities ...")
    face_centers = mesh.triangles_center
    k = min(FACE_SMOOTH_K, max(1, len(points)))
    tree = cKDTree(points)
    distances, indices = tree.query(face_centers, k=k)

    if k == 1:
        face_prob = samples[indices]
    else:
        distances = np.clip(distances, 1e-6, None)
        weights = 1.0 / distances
        face_prob = np.sum(samples[indices] * weights, axis=1) / np.sum(weights, axis=1)

    # Convert probability back to ambient occlusion value in [-1, 1] per derivation.
    face_ao = 2.0 * face_prob - 1.0

    if total_area > 0:
        mean_ao = float(np.average(face_ao, weights=area_faces))
    else:
        mean_ao = float("nan")

    # Maryam's theorem: C_M := 1 - ⟨AO⟩. Guard against NaNs if sampling failed.
    moeini_convexity = 1.0 - mean_ao if np.isfinite(mean_ao) else float("nan")

    print("=== Moeini Convexity Measure ===")
    print(f"Area-weighted mean AO:  {mean_ao:.6f}")
    print(f"C_M = 1 - <AO>:         {moeini_convexity:.6f}")
    if np.isfinite(moeini_convexity) and total_area > 0:
        # Analytical prediction from Maryam's theorem.
        expected_shadow = (total_area * moeini_convexity) / 4.0
        print(f"S * C_M / 4:           {expected_shadow:.6f}")
        print(f"S / 4:                 {total_area / 4.0:.6f}")

        print("Estimating average shadow via Monte Carlo ...")
        # Independent Monte Carlo validation of the projected-area prediction.
        shadow_mean = monte_carlo_shadow(mesh, rotations=args.rotations, rng=rng)
        print(f"Monte Carlo mean shadow: {shadow_mean:.6f}")
        if expected_shadow > 0:
            print(f"MC / (S*C_M/4):       {shadow_mean / expected_shadow:.6f}")
        if total_area > 0:
            print(f"MC / (S/4):           {shadow_mean / (total_area / 4.0):.6f}")


if __name__ == "__main__":
    main()

