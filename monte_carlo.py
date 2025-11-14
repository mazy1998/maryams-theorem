import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time

# Utility ---------------------------------------------------------------------

def set_axes_equal(ax, X, Y, Z):
    """Set 3D plot axes to equal scale for better geometry perception."""

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
R = 2.0  # Major radius
r = 1.0  # Minor radius

# Monte Carlo configuration
NUM_SAMPLES = 1000  # Rays per surface point
EPSILON = 1e-5


def detect_compute_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available torch device, preferring GPU backends."""
    if prefer_gpu:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = detect_compute_device()
TORCH_DTYPE = torch.float32 if DEVICE.type == "mps" else torch.float64
TOL = 1e-4 if TORCH_DTYPE == torch.float32 else 1e-8
ROOT_IMAG_TOL = 1e-4 if TORCH_DTYPE == torch.float32 else 1e-10
torch.manual_seed(1234)

# Function to get point on torus from parameters u,v (u: 0-2pi azimuthal, v: 0-2pi poloidal)
def torus_point(u, v):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.array([x, y, z])

# Implicit equation of torus f(X) = 0 (NumPy helper)
def torus_implicit(X):
    x, y, z = X
    return (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2


def torus_implicit_torch(points: torch.Tensor) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    R_t = torch.as_tensor(R, dtype=points.dtype, device=points.device)
    r_t = torch.as_tensor(r, dtype=points.dtype, device=points.device)
    return (torch.sqrt(x * x + y * y) - R_t) ** 2 + z * z - r_t * r_t


def sample_unit_vectors(num_samples: int, device: torch.device, dtype: torch.dtype = TORCH_DTYPE) -> torch.Tensor:
    vectors = torch.randn((num_samples, 3), device=device, dtype=dtype)
    return vectors / torch.clamp(vectors.norm(dim=-1, keepdim=True), min=1e-12)


@torch.no_grad()
def ray_intersects_torus_batch(
    P: torch.Tensor,
    directions: torch.Tensor,
    epsilon: float = EPSILON,
    tol: float = TOL,
) -> torch.Tensor:
    """Vectorized ray/torus intersection test for a batch of directions."""

    device = directions.device
    dtype = directions.dtype

    P = P.to(device=device, dtype=dtype)
    px, py, pz = P.unbind()
    vx = directions[:, 0]
    vy = directions[:, 1]
    vz = directions[:, 2]

    dotVV = torch.sum(directions * directions, dim=-1)
    dotPV = px * vx + py * vy + pz * vz
    dotPP_val = torch.sum(P * P)

    dotPV_xy = px * vx + py * vy
    dotPP_xy_val = px * px + py * py
    dotVV_xy = vx * vx + vy * vy

    R_t = torch.as_tensor(R, dtype=dtype, device=device)
    r_t = torch.as_tensor(r, dtype=dtype, device=device)

    alpha = dotVV
    beta = 2 * dotPV
    gamma = torch.full_like(alpha, dotPP_val - r_t * r_t + R_t * R_t)

    delta = dotVV_xy
    eps_coeff = 2 * dotPV_xy
    zeta = torch.full_like(alpha, dotPP_xy_val)

    a4 = alpha * alpha
    a3 = 2 * alpha * beta
    a2 = 2 * alpha * gamma + beta * beta - 4 * R_t * R_t * delta
    a1 = 2 * beta * gamma - 4 * R_t * R_t * eps_coeff
    a0 = gamma * gamma - 4 * R_t * R_t * zeta

    coeffs = torch.stack((a4, a3, a2, a1, a0), dim=-1)
    valid_mask = torch.abs(coeffs[:, 0]) > 1e-12
    if not bool(valid_mask.any()):
        return torch.zeros(directions.shape[0], dtype=torch.bool, device=device)

    coeffs_valid = coeffs[valid_mask]
    monic = coeffs_valid / coeffs_valid[:, :1]

    c3 = monic[:, 1]
    c2 = monic[:, 2]
    c1 = monic[:, 3]
    c0 = monic[:, 4]

    comp = torch.zeros((coeffs_valid.shape[0], 4, 4), dtype=monic.dtype, device=monic.device)
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    comp[:, 3, 2] = 1.0
    comp[:, 0, 3] = -c0
    comp[:, 1, 3] = -c1
    comp[:, 2, 3] = -c2
    comp[:, 3, 3] = -c3

    if comp.device.type == "mps":
        roots = torch.linalg.eigvals(comp.cpu())
        roots_real = roots.real.to(device)
        roots_imag = roots.imag.to(device)
    else:
        roots = torch.linalg.eigvals(comp)
        roots_real = roots.real
        roots_imag = roots.imag

    valid_roots_mask = (torch.abs(roots_imag) < ROOT_IMAG_TOL) & (roots_real > epsilon)
    has_valid_roots = valid_roots_mask.any(dim=-1)

    candidate_vals = torch.where(
        valid_roots_mask,
        roots_real,
        torch.full_like(roots_real, float("inf")),
    )
    t_min, _ = candidate_vals.min(dim=-1)
    has_positive_root = has_valid_roots & torch.isfinite(t_min)

    hits_valid = torch.zeros_like(has_positive_root)
    if bool(has_positive_root.any()):
        candidate_mask = has_positive_root
        t_selected = t_min[candidate_mask]
        directions_selected = directions[valid_mask][candidate_mask]
        X = P.unsqueeze(0) + t_selected.unsqueeze(-1) * directions_selected
        residual = torch.abs(torus_implicit_torch(X))
        hits_valid[candidate_mask] = residual < tol

    hits = torch.zeros(directions.shape[0], dtype=torch.bool, device=device)
    hits[valid_mask] = hits_valid
    return hits


def torus_normal(u, v):
    nx = np.cos(u) * np.cos(v)
    ny = np.sin(u) * np.cos(v)
    nz = np.sin(v)
    return np.array([nx, ny, nz])


@torch.no_grad()
def compute_prob(
    P: np.ndarray,
    normal: np.ndarray,
    num_samples: int = NUM_SAMPLES,
    directions: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> float:
    if device is None:
        device = directions.device if directions is not None else DEVICE

    if directions is None or directions.shape[0] != num_samples:
        dirs = sample_unit_vectors(num_samples, device=device, dtype=TORCH_DTYPE)
    else:
        if directions.device != device or directions.dtype != TORCH_DTYPE:
            dirs = directions.to(device=device, dtype=TORCH_DTYPE)
        else:
            dirs = directions

    normal_tensor = torch.as_tensor(normal, dtype=TORCH_DTYPE, device=device)
    normal_tensor = normal_tensor / torch.clamp(normal_tensor.norm(), min=1e-12)

    dot = torch.matmul(dirs, normal_tensor)
    dirs_adjusted = torch.where(dot.unsqueeze(-1) >= 0, dirs, -dirs)

    P_tensor = torch.as_tensor(P, dtype=TORCH_DTYPE, device=device)
    hits = ray_intersects_torus_batch(P_tensor, dirs_adjusted, epsilon=EPSILON, tol=TOL)
    return 0.5 + 0.5 * hits.float().mean().item()

# Grid for surface
num_u = 20
num_v = 20
u = np.linspace(0, 2*np.pi, num_u)
v = np.linspace(0, 2*np.pi, num_v)
U, V = np.meshgrid(u, v)

# Compute points and probs
X = np.zeros((num_u, num_v))
Y = np.zeros((num_u, num_v))
Z = np.zeros((num_u, num_v))
probs = np.zeros((num_u, num_v))

print(f"Using compute device: {DEVICE} (torch {torch.__version__})")
base_directions = sample_unit_vectors(NUM_SAMPLES, device=DEVICE, dtype=TORCH_DTYPE)
start_time = time.perf_counter()

for i in range(num_u):
    for j in range(num_v):
        P = torus_point(U[i,j], V[i,j])
        N = torus_normal(U[i,j], V[i,j])
        X[i,j] = P[0]
        Y[i,j] = P[1]
        Z[i,j] = P[2]
        probs[i,j] = compute_prob(
            P,
            N,
            num_samples=NUM_SAMPLES,
            directions=base_directions,
            device=DEVICE,
        )

elapsed = time.perf_counter() - start_time
print(f"Monte Carlo sweep completed in {elapsed:.2f}s using {NUM_SAMPLES} rays per point.")

# Calculate statistics
prob_flat = probs.flatten()
min_prob = np.min(prob_flat)
max_prob = np.max(prob_flat)
mean_prob = np.mean(prob_flat)
std_prob = np.std(prob_flat)
median_prob = np.median(prob_flat)

print("=== Ray Hit Probability Statistics ===")
print(f"Minimum probability: {min_prob:.4f}")
print(f"Maximum probability: {max_prob:.4f}")
print(f"Mean probability: {mean_prob:.4f}")
print(f"Standard deviation: {std_prob:.4f}")
print(f"Median probability: {median_prob:.4f}")
print(f"Range: {max_prob - min_prob:.4f}")
print(f"Total surface points analyzed: {len(prob_flat)}")

# Create visualization with two subplots
fig = plt.figure(figsize=(15, 6))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(probs), rstride=1, cstride=1, shade=False)

m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
m.set_array(probs)
fig.colorbar(m, ax=ax1, shrink=0.5, aspect=5, label='Probability')

ax1.set_title('Ray Hit Probability on Torus Surface')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
set_axes_equal(ax1, X, Y, Z)

# Histogram
ax2 = fig.add_subplot(122)
n_bins = 20
counts, bins, patches = ax2.hist(prob_flat, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')

# Color the histogram bars according to their probability values
for i, (count, bin_left, bin_right, patch) in enumerate(zip(counts, bins[:-1], bins[1:], patches)):
    bin_center = (bin_left + bin_right) / 2
    normalized_prob = (bin_center - min_prob) / (max_prob - min_prob) if max_prob > min_prob else 0
    patch.set_facecolor(plt.cm.viridis(normalized_prob))

ax2.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.3f}')
ax2.axvline(median_prob, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_prob:.3f}')
ax2.axvline(min_prob, color='blue', linestyle=':', linewidth=1, label=f'Min: {min_prob:.3f}')
ax2.axvline(max_prob, color='green', linestyle=':', linewidth=1, label=f'Max: {max_prob:.3f}')

ax2.set_xlabel('Ray Hit Probability')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Ray Hit Probabilities')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()