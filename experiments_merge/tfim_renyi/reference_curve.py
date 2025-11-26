import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising

# === CONFIGURATION ===
N_QUBITS = 16
MAX_SUBSYSTEM = N_QUBITS // 2  # Calculates l=1..8
H_VALUES = [0.8, 1.0, 1.2]     # The specific values requested
J_VAL = 1.0                    # NetKet J=1.0 => Ferromagnetic (-1 * ZZ)
OUTPUT_FILE = "tfim_16_renyi_ref.csv"

def build_tfim_hamiltonian(N: int, h: float, J: float) -> Ising:
    # pbc=True (Ring) to match reference entropy magnitudes
    graph = Hypercube(length=N, n_dim=1, pbc=True)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    return Ising(hilbert, graph, h=h, J=J)

def get_ground_state(ham: Ising):
    # Compute the single lowest eigenvector
    vals, vecs = eigsh(ham.to_sparse(), k=1, which="SA")
    return vals[0], vecs[:, 0]

def compute_renyi_entropy_s2(psi: np.ndarray, n_total: int, subsystem_size: int) -> float:
    """
    Computes S2 = -ln(Tr(rho^2)) using Schmidt decomposition (SVD).
    """
    dim_sub = 2**subsystem_size
    dim_env = 2**(n_total - subsystem_size)

    # reshape vector into matrix where rows = subsystem, cols = environment
    psi_matrix = psi.reshape((dim_sub, dim_env))

    # get singular values via SVD
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

    # eigenvalues of reduced density matrix are squares of singular values
    purity = np.sum(singular_values**4)

    return -np.log(purity)

def main():
    results = {}

    print(f"=== Starting Reference Calculation (ED) for N={N_QUBITS} ===")

    for h in H_VALUES:
        print(f"Processing h = {h} ...")

        ham = build_tfim_hamiltonian(N=N_QUBITS, h=h, J=J_VAL)
        _, psi = get_ground_state(ham)

        s2_curve = []
        for ell in range(1, MAX_SUBSYSTEM + 1):
            val = compute_renyi_entropy_s2(psi, N_QUBITS, ell)
            s2_curve.append(val)

        results[h] = s2_curve

    print(f"\nCalculation complete. Saving to {OUTPUT_FILE}...")

    header = "l,S2_h0.8,S2_h1.0,S2_h1.2"
    data_matrix = []
    for i, ell in enumerate(range(1, MAX_SUBSYSTEM + 1)):
        row = [ell]
        for h in H_VALUES:
            row.append(results[h][i])
        data_matrix.append(row)

    np.savetxt(
        OUTPUT_FILE,
        data_matrix,
        fmt=["%d", "%.8f", "%.8f", "%.8f"],
        delimiter=",",
        header=header,
        comments='' # Removes the '#' from the header line for cleaner CSV
    )

    print("Done. Plotting...")

    plt.figure(figsize=(7, 5), dpi=100)

    # Styles to match paper loosely
    styles = {
        0.8: {'marker': 's', 'color': 'tab:red', 'label': 'TFIM h=0.8'},
        1.0: {'marker': 'd', 'color': 'tab:blue', 'label': 'TFIM h=1.0'},
        1.2: {'marker': 'o', 'color': 'tab:green', 'label': 'TFIM h=1.2'}
    }

    x_axis = list(range(1, MAX_SUBSYSTEM + 1))

    for h in H_VALUES:
        plt.plot(x_axis, results[h],
                 marker=styles[h]['marker'],
                 linestyle='--',
                 color=styles[h]['color'],
                 label=styles[h]['label'],
                 linewidth=1.5, markersize=7)

    plt.xlabel(r"Subsystem size $\ell$", fontsize=14)
    plt.ylabel(r"Renyi Entropy $S_2$", fontsize=16)
    plt.title(f"Reference Curves (N={N_QUBITS}, PBC)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_axis)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()