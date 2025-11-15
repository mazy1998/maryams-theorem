"""Compute the Moeini Convexity Measure for a mesh and display the OBJ rotating."""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

import trimesh

import moeini_convexity_measure as mcm


def set_axes_equal(ax, verts: np.ndarray) -> None:
    verts = np.asarray(verts)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    spans = maxs - mins
    max_range = spans.max()
    if max_range <= 0:
        return
    centers = (mins + maxs) * 0.5
    half = max_range * 0.5
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def compute_convexity(
    mesh: trimesh.Trimesh,
    samples: int,
    rays: int,
    seed: int,
    compute_shadow: bool,
    rotations: int,
):
    rng = np.random.default_rng(seed)

    print(f"Sampling {samples} surface points ...")
    points, face_indices = trimesh.sample.sample_surface_even(mesh, samples)
    normals = mesh.face_normals[face_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print(f"Generating {rays} random rays ...")
    base_rays = mcm.sample_unit_vectors(rays, rng)

    print("Estimating cosine-weighted occlusion probabilities ...")
    samples_prob = mcm.estimate_occlusion(mesh, points, normals, base_rays, mcm.SURFACE_OFFSET)

    print("Interpolating per-face occlusion probabilities ...")
    face_centers = mesh.triangles_center
    k = min(mcm.FACE_SMOOTH_K, max(1, len(points)))
    tree = mcm.cKDTree(points)
    distances, indices = tree.query(face_centers, k=k)

    if k == 1:
        face_prob = samples_prob[indices]
    else:
        distances = np.clip(distances, 1e-6, None)
        weights = 1.0 / distances
        face_prob = np.sum(samples_prob[indices] * weights, axis=1) / np.sum(weights, axis=1)

    face_ao = 2.0 * face_prob - 1.0

    area_faces = mesh.area_faces
    total_area = float(area_faces.sum())
    if total_area > 0:
        mean_ao = float(np.average(face_ao, weights=area_faces))
    else:
        mean_ao = float("nan")

    cm_value = 1.0 - mean_ao if np.isfinite(mean_ao) else float("nan")
    predicted_shadow = (total_area * cm_value) / 4.0 if np.isfinite(cm_value) else float("nan")

    metrics = {
        "surface_area": total_area,
        "mean_ao": mean_ao,
        "cm": cm_value,
        "predicted_shadow": predicted_shadow,
        "face_ao": face_ao,
    }
    if compute_shadow and np.isfinite(predicted_shadow):
        print("Estimating average shadow via Monte Carlo ...")
        metrics["shadow_mc"] = mcm.monte_carlo_shadow(mesh, rotations=rotations, rng=rng)
    else:
        metrics["shadow_mc"] = None
    return metrics


def visualize_mesh(
    mesh: trimesh.Trimesh,
    face_ao: np.ndarray,
    frames: int,
    interval: int,
    output_dir: Path | None,
) -> None:
    verts_original = mesh.vertices
    faces = mesh.faces
    # Reorder coordinates so the Y axis becomes vertical (mapped to Matplotlib's Z).
    plot_verts = np.stack(
        (verts_original[:, 0], verts_original[:, 2], verts_original[:, 1]), axis=1
    )

    ao = np.asarray(face_ao, dtype=np.float64)
    ao_min = np.nanmin(ao)
    ao_max = np.nanmax(ao)
    if not np.isfinite(ao_min) or not np.isfinite(ao_max) or ao_max - ao_min <= 1e-8:
        colors = np.full(len(faces), 0.5)
    else:
        colors = (ao - ao_min) / (ao_max - ao_min)
    face_colors = plt.cm.viridis(np.clip(colors, 0.0, 1.0))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    tris = ax.plot_trisurf(
        plot_verts[:, 0],
        plot_verts[:, 1],
        plot_verts[:, 2],
        triangles=faces,
        edgecolor="none",
        alpha=1,
    )
    tris.set_facecolors(face_colors)
    set_axes_equal(ax, plot_verts)
    ax.set_title("Ambient Occlusion Shaded Mesh Rotation")
    ax.set_xlabel("X (original)")
    ax.set_ylabel("Z (original)")
    ax.set_zlabel("Y (vertical)")

    norm = plt.Normalize(vmin=np.nanmin(ao), vmax=np.nanmax(ao))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, label="Cosine-weighted AO")

    def update(frame: int):
        ax.view_init(elev=25, azim=frame * (360.0 / frames))
        return (tris,)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = 1000 / interval if interval > 0 else 20
        mp4_path = output_dir / "mesh_rotation.mp4"
        gif_path = output_dir / "mesh_rotation.gif"
        try:
            anim.save(mp4_path, writer="ffmpeg", fps=fps)
            print(f"Saved rotation video to {mp4_path}")
        except Exception as exc:
            print(f"Warning: failed to save MP4 ({exc})")
        try:
            gif_writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=gif_writer)
            print(f"Saved rotation GIF to {gif_path}")
        except Exception as exc:
            print(f"Warning: failed to save GIF ({exc})")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Moeini Convexity Measure and visualize the mesh rotating."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to the OBJ (or other trimesh-supported) file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=mcm.DEFAULT_SURFACE_SAMPLES,
        help="Number of surface samples for AO estimation.",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=mcm.DEFAULT_RAYS_PER_SAMPLE,
        help="Cosine-weighted hemisphere rays per surface sample.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=mcm.DEFAULT_ROTATIONS,
        help="Random orientations for Monte Carlo shadow verification.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=180,
        help="Number of frames for the rotation preview.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Delay between frames (ms) in the animation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store rotation MP4/GIF (video not saved if omitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )
    parser.add_argument(
        "--with-shadow",
        action="store_true",
        help="Also run the Monte Carlo shadow estimator (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mesh.exists():
        raise FileNotFoundError(f"Mesh not found at {args.mesh}")

    print(f"Loading mesh from {args.mesh} ...")
    mesh = mcm.load_mesh(args.mesh)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")

    metrics = compute_convexity(
        mesh=mesh,
        samples=args.samples,
        rays=args.rays,
        seed=args.seed,
        compute_shadow=args.with_shadow,
        rotations=args.rotations,
    )

    print("\n===== Moeini Convexity Summary =====")
    print(f"Surface area:           {metrics['surface_area']:.6f}")
    print(f"Area-weighted mean AO:  {metrics['mean_ao']:.6f}")
    print(f"C_M = 1 - <AO>:         {metrics['cm']:.6f}")
    print(f"S * C_M / 4:           {metrics['predicted_shadow']:.6f}")
    if metrics["shadow_mc"] is not None:
        shadow_mc = metrics["shadow_mc"]
        print(f"Monte Carlo shadow:     {shadow_mc:.6f}")
        if np.isfinite(metrics["predicted_shadow"]) and metrics["predicted_shadow"] > 0:
            print(
                f"Shadow ratio (MC / analytic): "
                f"{shadow_mc / metrics['predicted_shadow']:.6f}"
            )
        if metrics["surface_area"] > 0:
            print(f"Shadow ratio (MC / S/4): {shadow_mc / (metrics['surface_area'] / 4.0):.6f}")
    else:
        print("Monte Carlo shadow:     (skipped)")

    visualize_mesh(
        mesh,
        metrics["face_ao"],
        frames=args.frames,
        interval=args.interval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


