"""Plot the distribution of face areas for the Stanford bunny mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh


DEFAULT_BUNNY_PATH = Path(__file__).resolve().parent / "bunny.obj"
DEFAULT_BINS = 64


def load_bunny_mesh(path: Path) -> trimesh.Trimesh:
    """Load and sanitize the Stanford bunny mesh."""

    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.as_trimesh()

    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.rezero()
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bunny face area distribution")
    parser.add_argument(
        "--bunny",
        type=Path,
        default=DEFAULT_BUNNY_PATH,
        help="Path to the Stanford bunny OBJ file",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of histogram bins",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bunny.exists():
        raise FileNotFoundError(f"Bunny mesh not found at {args.bunny}")

    print(f"Loading bunny mesh from {args.bunny} ...")
    mesh = load_bunny_mesh(args.bunny)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)

    if face_areas.size == 0:
        raise RuntimeError("Loaded mesh contains no faces.")

    mean_area = face_areas.mean()
    median_area = np.median(face_areas)
    min_area = face_areas.min()
    max_area = face_areas.max()

    print(
        "Face area statistics (units^2): "
        f"count={face_areas.size}, "
        f"min={min_area:.6e}, "
        f"median={median_area:.6e}, "
        f"mean={mean_area:.6e}, "
        f"max={max_area:.6e}"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(face_areas, bins=args.bins, color="#4c72b0", edgecolor="white", linewidth=0.8)
    ax.set_title("Stanford Bunny Face Area Distribution")
    ax.set_xlabel("Face area (units^2)")
    ax.set_ylabel("Count")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

