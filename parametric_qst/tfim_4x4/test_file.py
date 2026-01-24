import sys
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- User Imports ---
# Assuming this script is run from the same location as your notebooks
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

# --- Constants ---
SIDE_LENGTH = 4
FILE_SAMPLE_COUNT = 20_000
TRAIN_SAMPLE_COUNT = 20_000

# Training Hyperparams
N_EPOCHS = 50
BATCH_SIZE = 1024
NUM_HIDDEN = 64
HYPER_NET_WIDTH = 64
K_STEPS = 10
GIBBS_NOISE_FRAC = 0.1
INIT_LR = 1e-2
FINAL_LR = 1e-4

# --- 1. Data Loading ---
h_support = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
file_names = [f"tfim_{SIDE_LENGTH}x{SIDE_LENGTH}_h{h:.2f}_{FILE_SAMPLE_COUNT}.npz" for h in h_support]
file_paths = [data_dir / fn for fn in file_names]

print(f"Loading data for fields: {h_support}")
samples_per_support = [TRAIN_SAMPLE_COUNT] * len(file_paths)

try:
    dataset = MeasurementDataset(file_paths, load_measurements_npz, ["h"], samples_per_support)
except FileNotFoundError:
    print("Error: Measurement files not found. Please ensure 'measurements/' directory exists.")
    sys.exit(1)

seed = 42
torch.manual_seed(seed)
rng = torch.Generator().manual_seed(seed)
loader = MeasurementLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=rng)

# --- 2. Model Initialization & Training ---
print("\n--- Initializing and Training Model ---")
model = SymmetricHyperRBM(num_v=dataset.num_qubits, num_h=NUM_HIDDEN,
                          hyper_dim=HYPER_NET_WIDTH, k=K_STEPS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
scheduler = get_sigmoid_curve(INIT_LR, FINAL_LR, N_EPOCHS*len(loader), 0.005)

model = train_loop(model, optimizer, loader, num_epochs=N_EPOCHS,
                   lr_schedule_fn=scheduler, noise_frac=GIBBS_NOISE_FRAC, rng=rng)

# --- 3. Sampling & Visualization (The Crosscheck) ---
print("\n--- Starting Distribution Crosscheck ---")

SAMPLES_EVAL = 5000 # Enough samples to see the distribution shape
K_STEPS_EVAL = 20   # Giving it a bit more mixing time for generation
RNG_EVAL = torch.Generator(device=device).manual_seed(123)
schedule_tensor = torch.tensor([1.0] * K_STEPS_EVAL, device=device, dtype=torch.float32)

# The fields you requested to visualize
check_fields = [1.0, 4.0, 7.0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle(f'Magnetization Distribution (Signed) for {SIDE_LENGTH}x{SIDE_LENGTH} TFIM', fontsize=16)

for i, h_val in enumerate(check_fields):
    print(f"Sampling for h = {h_val}...")

    # Create condition batch
    cond_batch = torch.full((SAMPLES_EVAL, 1), h_val, device=device, dtype=torch.float32)

    # Generate samples
    # Assuming generate returns raw bits (0s and 1s)
    samples = model.generate(cond_batch, T_schedule=schedule_tensor, rng=RNG_EVAL)

    # Convert 0/1 to +1/-1 spins
    # Map: 0 -> +1, 1 -> -1 (or vice versa, the distribution shape is what matters)
    spins_pm = 1.0 - 2.0 * samples.float()

    # Calculate signed magnetization per sample: sum(spins) / N
    mz_per_sample = spins_pm.mean(dim=1).cpu().numpy()

    # Plot Histogram
    ax = axes[i]
    # We use a fixed range -1 to 1 to compare easily
    n, bins, patches = ax.hist(mz_per_sample, bins=30, range=(-1.1, 1.1),
                               density=True, alpha=0.7, color='dodgerblue', edgecolor='black')

    # Styling
    ax.set_title(f"Field h = {h_val:.2f}", fontsize=14)
    ax.set_xlabel(r"Magnetization $M_z = \frac{1}{N}\sum \sigma_i^z$", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-1.1, 1.1)

    # Add mean marker
    mean_mz = np.mean(mz_per_sample)
    ax.axvline(mean_mz, color='red', linestyle='--', label=f"Mean: {mean_mz:.2f}")
    ax.legend()

axes[0].set_ylabel("Probability Density", fontsize=12)

plt.tight_layout()
output_filename = results_dir / "magnetization_distribution_check.png"
plt.savefig(output_filename)
print(f"\nAnalysis complete. Plot saved to: {output_filename}")
plt.show()