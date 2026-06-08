from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_handling import save_measurements_txt, save_state_txt


#### STATE CONSTRUCTION ####

def generate_vanilla_w_state(num_qubits: int) -> np.ndarray:
    state_dim = 1 << num_qubits

    state = np.zeros(state_dim, dtype=np.complex128)
    one_hot_indices = 1 << np.arange(num_qubits - 1, -1, -1, dtype=np.int64)
    state[one_hot_indices] = 1.0 / np.sqrt(num_qubits)

    return state


#### SAMPLE GENERATION ####

def sample_w_state_binary(
    num_qubits: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen_sites = rng.integers(low=0, high=num_qubits, size=num_samples)

    samples = np.zeros((num_samples, num_qubits), dtype=np.uint8)
    samples[np.arange(num_samples), chosen_sites] = 1

    return samples


def plot_site_frequencies(samples: np.ndarray) -> None:
    num_samples, num_qubits = samples.shape
    rel_freq = samples.sum(axis=0) / num_samples

    fig, ax = plt.subplots(figsize=(8, 4))
    sites = np.arange(1, num_qubits + 1)

    ax.bar(sites, rel_freq, width=0.8, edgecolor="black", alpha=0.7)
    ax.axhline(
        1.0 / num_qubits,
        color="red",
        linestyle="--",
        label=f"Expected = {1.0 / num_qubits:.3f}",
    )

    ax.set_xlabel("Spin-site index")
    ax.set_ylabel("Relative frequency of up-spin")
    ax.set_title("Training Data: Up-Spin Occurrence per Site")
    ax.set_xticks(sites)
    ax.set_ylim(0, rel_freq.max() * 1.2)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


#### RUN SCRIPT ####

if __name__ == "__main__":
    # edit parameters here
    rng_seed = 42
    num_qubits = 10
    num_samples = 50_000
    show_plot = True

    rng = np.random.default_rng(rng_seed)

    out_meas = Path("measurements")
    out_states = Path("state_vectors")
    out_meas.mkdir(parents=True, exist_ok=True)
    out_states.mkdir(parents=True, exist_ok=True)

    psi = generate_vanilla_w_state(num_qubits)
    samples = sample_w_state_binary(num_qubits, num_samples, rng)

    if show_plot:
        plot_site_frequencies(samples)

    state_header = {
        "system": "W_vanilla",
        "nqubits": int(num_qubits),
        "seed": int(rng_seed),
    }

    meas_header = {
        "basis": "Z" * num_qubits,
        "samples": int(num_samples),
        "seed": int(rng_seed),
    }

    state_path = out_states / f"w_vanilla_{num_qubits}_state.txt"
    meas_path = out_meas / f"w_vanilla_{num_qubits}_meas_{num_samples}.txt"

    save_state_txt(state_path, psi, {"state": state_header})
    save_measurements_txt(
        meas_path,
        samples,
        ["Z"] * num_qubits,
        {"state": state_header, "measurement": meas_header},
    )

    nonzero_indices = np.flatnonzero(np.abs(psi) > 0)[:10]
    print("First 10 amplitudes:")
    for idx in nonzero_indices:
        print(f"{idx:0{num_qubits}b}: {psi[idx]:.8f}")

    print(f"\nSize of state vector in memory: {psi.nbytes / (1024 ** 2):.2f} MB\n")
    print(f"Saved {psi.shape[0]} amplitudes (Re, Im) to {state_path}")
    print(f"Wrote {len(samples)} samples to {meas_path}")