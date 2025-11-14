"""Monte Carlo estimate of the average orthographic shadow area of the Stanford bunny.

This script samples uniformly from SO(3) (the space of rotations) using Haar
measure via random unit quaternions. For each orientation the bunny mesh is
projected onto the ground plane (XY) under an infinite light source aligned
with the negative Z direction. The shadow area is the measure of the union of
the projected mesh faces on that plane.

The average over many random orientations approximates the mean shadow area
with respect to the Haar measure on SO(3).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Configuration

DEFAULT_BUNNY_PATH = Path(__file__).resolve().parent / "teapot.obj"
DEFAULT_ROTATIONS = 512
TOL_TRIANGLE_AREA = 1e-12


# ---------------------------------------------------------------------------
# Helpers


def random_unit_quaternions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `count` quaternions uniformly from S³ (Haar measure on SO(3))."""

    raw = rng.normal(size=(count, 4))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.clip(norms, 1e-12, None)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a 3×3 rotation matrix."""

    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def project_shadow_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Return the area of the orthographic shadow onto the XY plane."""

    polygons = []
    for face in faces:
        tri = vertices[face, :2]
        area = 0.5 * abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )
        if area < TOL_TRIANGLE_AREA:
            continue
        polygons.append(Polygon(tri))

    if not polygons:
        return 0.0

    union = unary_union(polygons)
    return float(union.area)


def load_bunny(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.as_trimesh()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.rezero()
    return mesh


# ---------------------------------------------------------------------------
# Main routine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate average orthographic shadow area of the Stanford bunny."
    )
    parser.add_argument(
        "--bunny",
        type=Path,
        default=DEFAULT_BUNNY_PATH,
        help="Path to the bunny mesh (OBJ, STL, etc.).",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=DEFAULT_ROTATIONS,
        help="Number of random orientations sampled from SO(3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=20,
        help="Print progress every N samples (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bunny.exists():
        raise FileNotFoundError(f"Bunny mesh not found at {args.bunny}")

    print(f"Loading mesh from {args.bunny} ...")
    mesh = load_bunny(args.bunny)
    vertices = mesh.vertices
    faces = mesh.faces
    print(
        f"Loaded mesh with {len(vertices)} vertices, {len(faces)} faces, "
        f"watertight={mesh.is_watertight}"
    )
    surface_area = float(mesh.area)
    theoretical_mean = surface_area / 4.0

    rng = np.random.default_rng(args.seed)
    quats = random_unit_quaternions(args.rotations, rng)

    shadow_areas = np.empty(args.rotations, dtype=np.float64)
    start = time.perf_counter()

    for idx, quat in enumerate(quats, start=1):
        R = quaternion_to_matrix(quat)
        rotated = vertices @ R.T
        shadow_areas[idx - 1] = project_shadow_area(rotated, faces)

        if args.progress and idx % args.progress == 0:
            elapsed = time.perf_counter() - start
            print(f"  processed {idx}/{args.rotations} rotations (elapsed {elapsed:.2f}s)")

    elapsed = time.perf_counter() - start

    mean_area = shadow_areas.mean()
    std_area = shadow_areas.std(ddof=1)
    min_area = shadow_areas.min()
    max_area = shadow_areas.max()

    print("=== Average Shadow Estimate ===")
    print(f"Samples (rotations): {args.rotations}")
    print(f"Mean shadow area:    {mean_area:.6f}")
    print(f"Std. deviation:      {std_area:.6f}")
    print(f"Min / Max area:      {min_area:.6f} / {max_area:.6f}")
    print(f"Computation time:    {elapsed:.2f}s")
    print()
    print("=== Surface Area Relationship ===")
    print(f"Surface area (A):        {surface_area:.6f}")
    print(f"Theoretical A/4:         {theoretical_mean:.6f}")
    if theoretical_mean > 0:
        ratio = mean_area / theoretical_mean
    else:
        ratio = float('nan')
    print(f"Mean shadow / (A/4):     {ratio:.6f}")
    print(f"Difference mean - A/4:   {mean_area - theoretical_mean:.6f}")


if __name__ == "__main__":
    main()

