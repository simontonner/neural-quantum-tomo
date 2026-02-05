import sys
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- User Imports ---
sys.path.append(str(Path("..").resolve()))

try:
    from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader
    from hyper_rbm import SymmetricHyperRBM, train_loop, get_sigmoid_curve
except ImportError as e:
    print("Error: Could not import project modules. Make sure 'data_handling.py' and 'hyper_rbm.py' are in the parent directory.")
    sys.exit(1)

# --- Configuration ---
data_dir = Path("measurements")
models_dir = Path("models")
results_dir = Path("results")
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# --- Constants for 1D Chain ---
SYSTEM_SIZE = 16
FILE_SAMPLE_COUNT = 20_000
TRAIN_SAMPLE_COUNT = 20_000

# Training Hyperparams
N_EPOCHS = 50
BATCH_SIZE = 1024
NUM_HIDDEN = 64
HYPER_NET_WIDTH = 64
K_STEPS = 20
GIBBS_NOISE_FRAC = 0.1
INIT_LR = 1e-2
FINAL_LR = 1e-4

# --- 1. Data Loading ---
h_support = [0.50, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50]

file_names = [f"tfim_{SYSTEM_SIZE}_h{h:.2f}_{FILE_SAMPLE_COUNT}.npz" for h in h_support]
file_paths = [data_dir / fn for fn in file_names]

print(f"Loading data for fields: {h_support}")
samples_per_support = [TRAIN_SAMPLE_COUNT] * len(file_paths)

try:
    dataset = MeasurementDataset(file_paths, load_measurements_npz, ["h"], samples_per_support)
except FileNotFoundError:
    print(f"Error: Measurement files not found in {data_dir}. Please check path.")
    sys.exit(1)

seed = 42
torch.manual_seed(seed)
rng = torch.Generator().manual_seed(seed)

loader = MeasurementLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=rng)

# --- 2. Model Initialization & Training ---
print("\n--- Initializing and Training Model (1D Chain) ---")
model = SymmetricHyperRBM(num_v=SYSTEM_SIZE, num_h=NUM_HIDDEN,
                          hyper_dim=HYPER_NET_WIDTH, k=K_STEPS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = get_sigmoid_curve(INIT_LR, FINAL_LR, N_EPOCHS*len(loader), 0.005)

model = train_loop(model, optimizer, loader, num_epochs=N_EPOCHS,
                   lr_schedule_fn=scheduler, noise_frac=GIBBS_NOISE_FRAC, rng=rng)

# --- 3. Sampling & Visualization (Fixed Bins) ---
print("\n--- Starting Distribution Crosscheck ---")

check_fields = [0.5, 1.15, 1.5]

SAMPLES_EVAL = 5000
K_STEPS_EVAL = 50
RNG_EVAL = torch.Generator(device=device).manual_seed(123)
schedule_tensor = torch.tensor([1.0] * K_STEPS_EVAL, device=device, dtype=torch.float32)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle(f'Real Magnetization Distribution for 1D TFIM (N={SYSTEM_SIZE})', fontsize=16)

# --- VISUALIZATION FIX ---
# There are exactly N+1 possible magnetization values.
# Step size = 2/N.
# We define bin edges such that the bin center aligns with the integer steps.
delta = 2.0 / SYSTEM_SIZE
half_delta = delta / 2.0
# Create edges shifted by half a step to center the bars
bin_edges = np.linspace(-1.0 - half_delta, 1.0 + half_delta, SYSTEM_SIZE + 2)

for i, h_val in enumerate(check_fields):
    print(f"Sampling for h = {h_val}...")

    cond_batch = torch.full((SAMPLES_EVAL, 1), h_val, device=device, dtype=torch.float32)
    samples = model.generate(cond_batch, T_schedule=schedule_tensor, rng=RNG_EVAL)

    # 1 -> +1, 0 -> -1 mapping
    spins_pm = 1.0 - 2.0 * samples.float()
    mz_per_sample = spins_pm.mean(dim=1).cpu().numpy()

    ax = axes[i]

    # Plot using fixed bin edges
    ax.hist(mz_per_sample, bins=bin_edges, density=True,
            alpha=0.75, color='forestgreen', edgecolor='black', linewidth=1.0)

    ax.set_title(f"Field h = {h_val:.2f}", fontsize=14)
    ax.set_xlabel(r"Magnetization $M_z$", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)

    mean_mz = np.mean(mz_per_sample)
    ax.axvline(mean_mz, color='red', linestyle='--', label=f"Mean: {mean_mz:.2f}")
    ax.legend()

axes[0].set_ylabel("Probability Density", fontsize=12)

plt.tight_layout()
output_filename = results_dir / "magnetization_1d_check_fixed.png"
plt.savefig(output_filename)
print(f"\nAnalysis complete. Plot saved to: {output_filename}")
plt.show()