"""Monte Carlo ray–torus intersection simulation backed by MLX kernels.

This script mirrors the behaviour of `monte_carlo.py` but rewrites the
ray-intersection core so every heavy computation runs in MLX on Apple GPUs.

- Directions are sampled once per sweep on the GPU.
- The quartic intersection is solved with a batched Ferrari method expressed
  entirely in MLX primitives, keeping data on-device.
- Surface points are processed in batches to balance memory and throughput.

The output matches the original script: statistics and a 3D surface coloured by
hit probability plus a histogram of the probability distribution.
"""

from __future__ import annotations

import time
import matplotlib.pyplot as plt
import numpy as np

import mlx.core as mx


def set_axes_equal(ax, X, Y, Z):
    """Set 3D plot axes to equal scale."""

    x_vals = X.flatten()
    y_vals = Y.flatten()
    z_vals = Z.flatten()

    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    z_range = z_vals.max() - z_vals.min()

    max_range = max(x_range, y_range, z_range)
    if max_range == 0:
        return

    x_mid = (x_vals.max() + x_vals.min()) * 0.5
    y_mid = (y_vals.max() + y_vals.min()) * 0.5
    z_mid = (z_vals.max() + z_vals.min()) * 0.5

    half = max_range * 0.5

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


# Torus parameters
R_MAJOR = 1.5
R_MINOR = 1.0


# Monte Carlo configuration
NUM_SAMPLES = 20_000  # rays per surface point
EPSILON = 1e-5
TOL = 1e-4
BATCH_SIZE = 2048  # number of surface points per MLX batch


def configure_device() -> mx.Device:
    """Select the GPU device when available."""

    try:  # Prefer the GPU backend on Apple silicon
        mx.set_default_device(mx.gpu)
    except Exception:  # pragma: no cover - falls back to current default
        pass
    return mx.default_device()


DEVICE = configure_device()


def torus_implicit_mx(points: mx.array) -> mx.array:
    """Evaluate the torus implicit function f(X) for MLX arrays."""

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    radial = mx.sqrt(mx.maximum(x * x + y * y, 0.0))
    return (radial - R_MAJOR) ** 2 + z * z - R_MINOR**2


def sample_unit_vectors(num_samples: int) -> mx.array:
    """Draw random unit vectors on the sphere using MLX."""

    vecs = mx.random.normal((num_samples, 3))
    norms = mx.sqrt(mx.maximum(mx.sum(vecs * vecs, axis=-1, keepdims=True), 1e-12))
    return vecs / norms


def solve_quartic(coeffs: mx.array) -> mx.array:
    """Solve a batch of quartic equations using Ferrari's method in MLX.

    Args:
        coeffs: array with shape (..., 5) encoding [a4, a3, a2, a1, a0].

    Returns:
        roots: array with shape (..., 4) of the quartic roots (real-valued).
    """

    a4 = coeffs[..., 0]
    a3 = coeffs[..., 1]
    a2 = coeffs[..., 2]
    a1 = coeffs[..., 3]
    a0 = coeffs[..., 4]

    inv_a4 = 1.0 / a4
    a = a3 * inv_a4
    b = a2 * inv_a4
    c = a1 * inv_a4
    d = a0 * inv_a4

    alpha = -3.0 * a * a / 8.0 + b
    beta = a * a * a / 8.0 - a * b / 2.0 + c
    gamma = -3.0 * a * a * a * a / 256.0 + a * a * b / 16.0 - a * c / 4.0 + d

    P = -alpha * alpha / 12.0 - gamma
    Q = -alpha * alpha * alpha / 108.0 + alpha * gamma / 3.0 - (beta * beta) / 8.0

    sqrt_inner = mx.maximum(Q * Q / 4.0 + P * P * P / 27.0, 0.0)
    R = -Q / 2.0 + mx.sqrt(sqrt_inner)

    # Robust cube root that preserves the sign for negative values
    U = mx.sign(R) * mx.power(mx.maximum(mx.abs(R), 1e-18), 1.0 / 3.0)
    U = mx.where(mx.abs(U) < 1e-6, 0.0, U)

    y = -5.0 * alpha / 6.0 - U
    y = y - mx.where(mx.abs(U) > 1e-6, P / (3.0 * U), 0.0)

    W_sq = mx.maximum(alpha + 2.0 * y, 0.0)
    W = mx.sqrt(W_sq)

    abs_W = mx.abs(W)
    two_beta_over_W = mx.where(abs_W > 1e-6, 2.0 * beta / W, 0.0)

    term_base = -3.0 * alpha - 2.0 * y
    term1 = mx.maximum(term_base + two_beta_over_W, 0.0)
    term2 = mx.maximum(term_base - two_beta_over_W, 0.0)

    sqrt_term1 = mx.sqrt(term1)
    sqrt_term2 = mx.sqrt(term2)

    root1 = -a / 4.0 + 0.5 * (W + sqrt_term1)
    root2 = -a / 4.0 + 0.5 * (W - sqrt_term1)
    root3 = -a / 4.0 + 0.5 * (-W + sqrt_term2)
    root4 = -a / 4.0 + 0.5 * (-W - sqrt_term2)

    return mx.stack((root1, root2, root3, root4), axis=-1)


def ray_hits_batch(points: mx.array, directions: mx.array) -> mx.array:
    """Compute intersection hits for a batch of surface points.

    Args:
        points: MLX array with shape (B, 3).
        directions: MLX array with shape (S, 3) or (B, S, 3).

    Returns:
        Boolean MLX array with shape (B, S) marking intersections.
    """

    B = points.shape[0]
    if directions.ndim == 2:
        dirs = mx.expand_dims(directions, axis=0)
    else:
        dirs = directions
    S = dirs.shape[1]

    pts = mx.expand_dims(points, axis=1)

    dotVV = mx.sum(dirs * dirs, axis=-1)
    dotPV = mx.sum(pts * dirs, axis=-1)
    dotPP = mx.sum(points * points, axis=-1, keepdims=True)

    dir_xy = dirs[..., :2]
    pt_xy = mx.expand_dims(points[:, :2], axis=1)
    dotPV_xy = mx.sum(pt_xy * dir_xy, axis=-1)
    dotPP_xy = mx.sum(points[:, :2] * points[:, :2], axis=-1, keepdims=True)
    dotVV_xy = mx.sum(dir_xy * dir_xy, axis=-1)

    R_sq = R_MAJOR * R_MAJOR

    alpha = dotVV + mx.zeros_like(dotPV)
    beta = 2.0 * dotPV
    gamma = dotPP + (R_sq - R_MINOR * R_MINOR)
    gamma = gamma + mx.zeros_like(dotPV)

    delta = dotVV_xy + mx.zeros_like(dotPV)
    eps_coeff = 2.0 * dotPV_xy
    zeta = dotPP_xy + mx.zeros_like(dotPV)

    a4 = alpha * alpha
    a3 = 2.0 * alpha * beta
    a2 = 2.0 * alpha * gamma + beta * beta - 4.0 * R_sq * delta
    a1 = 2.0 * beta * gamma - 4.0 * R_sq * eps_coeff
    a0 = gamma * gamma - 4.0 * R_sq * zeta

    coeffs = mx.stack((a4, a3, a2, a1, a0), axis=-1)
    coeffs_flat = mx.reshape(coeffs, (-1, 5))

    roots = solve_quartic(coeffs_flat)
    roots = mx.reshape(roots, (B, S, 4))

    positive_mask = roots > EPSILON
    a4_e = mx.expand_dims(a4, axis=-1)
    a3_e = mx.expand_dims(a3, axis=-1)
    a2_e = mx.expand_dims(a2, axis=-1)
    a1_e = mx.expand_dims(a1, axis=-1)
    a0_e = mx.expand_dims(a0, axis=-1)

    t_refined = mx.where(positive_mask, roots, 0.0)
    for _ in range(3):
        f_val = (((a4_e * t_refined + a3_e) * t_refined + a2_e) * t_refined + a1_e) * t_refined + a0_e
        df_val = ((4.0 * a4_e * t_refined + 3.0 * a3_e) * t_refined + 2.0 * a2_e) * t_refined + a1_e
        df_safe = mx.where(mx.abs(df_val) > 1e-6, df_val, 1.0)
        t_refined = t_refined - mx.where(positive_mask, f_val / df_safe, 0.0)

    valid_refined = mx.logical_and(positive_mask, t_refined > EPSILON)

    pts_exp = mx.reshape(pts, (B, 1, 1, 3))
    dirs_exp = mx.expand_dims(dirs, axis=2)
    t_exp = mx.expand_dims(t_refined, axis=-1)

    X_candidates = pts_exp + t_exp * dirs_exp
    residual = mx.abs(torus_implicit_mx(X_candidates))

    hit_candidates = mx.logical_and(valid_refined, residual < TOL)
    hits = mx.any(hit_candidates, axis=-1)
    return hits


def compute_probabilities(points: mx.array, normals: mx.array, directions: mx.array) -> mx.array:
    """Estimate hit probabilities for each surface point."""

    probs = []
    num_points = points.shape[0]
    sample_count = directions.shape[0]

    dirs_expanded = mx.expand_dims(directions, axis=0)

    for start in range(0, num_points, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_points)
        batch = points[start:end]
        normals_batch = normals[start:end]

        normals_exp = mx.expand_dims(normals_batch, axis=1)
        dot = mx.sum(normals_exp * dirs_expanded, axis=-1)
        dirs_adjusted = mx.where(
            mx.expand_dims(dot >= 0, axis=-1),
            dirs_expanded,
            -dirs_expanded,
        )

        hits = ray_hits_batch(batch, dirs_adjusted)
        hit_counts = mx.sum(mx.where(hits, 1.0, 0.0), axis=1)
        probs.append(0.5 + 0.5 * (hit_counts / sample_count))

    return mx.concatenate(probs, axis=0)


def main() -> None:
    print(f"Using MLX device: {DEVICE}")

    directions = sample_unit_vectors(NUM_SAMPLES)
    mx.eval(directions)

    num_u = 100
    num_v = 100
    u = np.linspace(0.0, 2.0 * np.pi, num_u)
    v = np.linspace(0.0, 2.0 * np.pi, num_v)
    U, V = np.meshgrid(u, v, indexing="ij")

    X = (R_MAJOR + R_MINOR * np.cos(V)) * np.cos(U)
    Y = (R_MAJOR + R_MINOR * np.cos(V)) * np.sin(U)
    Z = R_MINOR * np.sin(V)

    points_np = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    normals_np = np.stack(
        (np.cos(U) * np.cos(V), np.sin(U) * np.cos(V), np.sin(V)), axis=-1
    ).reshape(-1, 3)

    points = mx.array(points_np)
    normals = mx.array(normals_np)

    start = time.perf_counter()
    probs = compute_probabilities(points, normals, directions)
    mx.eval(probs)
    elapsed = time.perf_counter() - start

    print(
        f"Monte Carlo sweep completed in {elapsed:.2f}s with {NUM_SAMPLES} rays per point"
    )

    probs_np = np.array(probs).reshape(num_u, num_v)
    prob_flat = probs_np.flatten()

    min_prob = float(prob_flat.min())
    max_prob = float(prob_flat.max())
    mean_prob = float(prob_flat.mean())
    std_prob = float(prob_flat.std())
    median_prob = float(np.median(prob_flat))

    print("=== Ray Hit Probability Statistics ===")
    print(f"Minimum probability: {min_prob:.4f}")
    print(f"Maximum probability: {max_prob:.4f}")
    print(f"Mean probability: {mean_prob:.4f}")
    print(f"Standard deviation: {std_prob:.4f}")
    print(f"Median probability: {median_prob:.4f}")
    print(f"Range: {max_prob - min_prob:.4f}")
    print(f"Total surface points analysed: {prob_flat.size}")

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(
        X,
        Y,
        Z,
        facecolors=plt.cm.viridis(probs_np),
        rstride=1,
        cstride=1,
        shade=False,
    )

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(probs_np)
    fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=5, label="Probability")

    ax1.set_title("Ray Hit Probability on Torus Surface (MLX)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    set_axes_equal(ax1, X, Y, Z)

    ax2 = fig.add_subplot(122)
    n_bins = 30
    counts, bins, patches = ax2.hist(
        prob_flat,
        bins=n_bins,
        alpha=0.75,
        color="skyblue",
        edgecolor="black",
    )

    for (count, bin_left, bin_right, patch) in zip(counts, bins[:-1], bins[1:], patches):
        bin_center = 0.5 * (bin_left + bin_right)
        normalized = 0.0 if max_prob == min_prob else (bin_center - min_prob) / (max_prob - min_prob)
        patch.set_facecolor(plt.cm.viridis(normalized))

    ax2.axvline(mean_prob, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_prob:.3f}")
    ax2.axvline(median_prob, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_prob:.3f}")
    ax2.axvline(min_prob, color="blue", linestyle=":", linewidth=1, label=f"Min: {min_prob:.3f}")
    ax2.axvline(max_prob, color="green", linestyle=":", linewidth=1, label=f"Max: {max_prob:.3f}")

    ax2.set_xlabel("Ray Hit Probability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Ray Hit Probabilities (MLX)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

