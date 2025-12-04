import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

import netket as nk
from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising

# === SYSTEM CONFIGURATION ===
N_QUBITS = 16
SIDE_LENGTH = N_QUBITS  # 1D Chain
PBC = True              # Periodic Boundary Conditions

# Physics Parameters
# h=0.8 (Ferromagnetic), h=1.0 (Critical), h=1.2 (Paramagnetic)
H_VALUES = [0.8, 1.0, 1.2]
J_VAL = 1.0

# Max subsystem size to calculate (N/2)
MAX_SUBSYSTEM = N_QUBITS // 2

OUTPUT_FILE = "tfim_1d_renyi_entropy.csv"

def build_hamiltonian(h_field):
    """
    Constructs the 1D TFIM Hamiltonian: H = -J * sum(ZZ) - h * sum(X)
    """
    graph = Hypercube(length=SIDE_LENGTH, n_dim=1, pbc=PBC)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    return Ising(hilbert, graph=graph, h=h_field, J=J_VAL)

def get_ground_state(hamiltonian):
    """
    Computes the exact ground state using sparse diagonalization (Lanczos/Arnoldi).
    """
    sp_mat = hamiltonian.to_sparse()
    # 'SA' = Smallest Algebraic eigenvalues
    vals, vecs = eigsh(sp_mat, k=1, which="SA")
    return vals[0], vecs[:, 0]

def compute_renyi_entropy_s2(psi, n_total, l_sub):
    """
    Computes S2 = -ln(Tr(rho^2)) via Schmidt Decomposition (SVD).

    1. Reshape vector psi into a matrix (dim_subsystem x dim_environment).
    2. Compute Singular Values (s_i).
    3. The eigenvalues of the Reduced Density Matrix are s_i^2.
    4. Tr(rho^2) = sum((s_i^2)^2) = sum(s_i^4).
    """
    dim_sub = 2**l_sub
    dim_env = 2**(n_total - l_sub)

    # Note: NetKet's basis is lexicographic, so simple reshaping works
    # for a contiguous block of qubits [0, ..., l-1].
    psi_matrix = psi.reshape((dim_sub, dim_env))

    # We only need singular values, not the unitary matrices
    singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

    purity = np.sum(singular_values**4)

    # Clip to avoid log(0) in case of numerical precision issues
    return -np.log(np.maximum(purity, 1e-15))

# === MAIN EXECUTION ===

print(f"{'='*60}")
print(f"1D TFIM RENYI ENTROPY CALCULATION")
print(f"System: N={N_QUBITS} Spins, Periodic Boundaries={PBC}")
print(f"{'='*60}")

results = {}

for h in H_VALUES:
    print(f"Processing transverse field h = {h}...")

    # 1. Build Operator
    ham = build_hamiltonian(h)

    # 2. Solve GS
    _, psi = get_ground_state(ham)

    # 3. Compute Entropy for subsystem sizes l=1 to N/2
    s2_curve = []
    for l in range(1, MAX_SUBSYSTEM + 1):
        val = compute_renyi_entropy_s2(psi, N_QUBITS, l)
        s2_curve.append(val)

    results[h] = s2_curve

print("-" * 60)
print(f"Calculation complete. Saving to {OUTPUT_FILE}...")

# Prepare data for CSV export
# Format: l, S2_h1, S2_h2, ...
header_str = "l," + ",".join([f"S2_h{h}" for h in H_VALUES])
data_rows = []

for i, l in enumerate(range(1, MAX_SUBSYSTEM + 1)):
    row = [l]
    for h in H_VALUES:
        row.append(results[h][i])
    data_rows.append(row)

np.savetxt(
    OUTPUT_FILE,
    data_rows,
    fmt=["%d"] + ["%.8f"]*len(H_VALUES),
    delimiter=",",
    header=header_str,
    comments=''
)

# === PLOTTING ===

plt.figure(figsize=(7, 5), dpi=100)

styles = {
    0.8: {'marker': 's', 'color': 'tab:red',   'label': r'Ferromagnetic ($h=0.8$)'},
    1.0: {'marker': 'd', 'color': 'tab:blue',  'label': r'Critical ($h=1.0$)'},
    1.2: {'marker': 'o', 'color': 'tab:green', 'label': r'Paramagnetic ($h=1.2$)'}
}

l_axis = list(range(1, MAX_SUBSYSTEM + 1))

for h in H_VALUES:
    plt.plot(l_axis, results[h],
             marker=styles[h]['marker'],
             color=styles[h]['color'],
             linestyle='--',
             linewidth=1.5,
             markersize=7,
             label=styles[h]['label'])

plt.xlabel(r"Subsystem size $\ell$", fontsize=12)
plt.ylabel(r"Renyi Entropy $S_2$", fontsize=12)
plt.title(f"TFIM Ground State Entanglement ($N={N_QUBITS}$)", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(l_axis)
plt.tight_layout()

plt.show()