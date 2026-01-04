import os
import time
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"         # cpu runs faster. at least on my machine
os.environ["JAX_ENABLE_X64"] = "1"          # use 64-bit floats for higher precision

import jax
import netket as nk
from scipy.sparse.linalg import eigsh


#### STATE CONSTRUCTION VIA NETKET ####

def build_heisenberg(L: int, *, pbc: bool = True):
    g = nk.graph.Grid(extent=[L, L], pbc=pbc)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = nk.operator.Heisenberg(hilbert=hi, graph=g)
    return hi, g, H


def calculate_groundstate(H) -> tuple[float, np.ndarray]:
    start = time.time()
    # k=1 tells Lanczos solver stops once the lowest eigenpair has converged
    # SA: smallest algebraic puts ground state first
    evals, evecs = eigsh(H.to_sparse(), k=1, which='SA')
    duration = time.time() - start
    energy = float(evals[0])
    psi = np.array(evecs[:, 0], dtype=np.complex128, copy=False)
    print(f"Diagonalization took {duration:.3f}s. Selected ground state at index 0 with energy {energy:.8f}.")
    return energy, psi


def main():
    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("NetKet version:", nk.__version__)

    #### define the hamiltonian ####

    L = 4                      # 4x4 is the max we solve exactly below
    hi, g, ha = build_heisenberg(L, pbc=True)

    #### use a GCNN ansatz with symmetries ####

    symms = g.automorphisms()
    ma = nk.models.GCNN(symmetries=symms, layers=4, features=16, complex_output=True)

    #### optimize with ADAM and natural gradient (SR) ####

    opt = nk.optimizer.Adam(learning_rate=0.005)
    sr = nk.optimizer.SR(diag_shift=0.01)     # let NetKet choose its default linear solver (CG-like)

    #### sampler and variational state driver ####

    # use many chains and large chunk_size to keep JAX kernels fat on CPU.
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, n_chains=512, sweep_size=2)
    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=8192,
        chunk_size=8192,
        n_discard_per_chain=64,
        seed=123,
    )

    vmc = nk.driver.VMC(hamiltonian=ha, optimizer=opt, variational_state=vs, preconditioner=sr)

    start = time.time()
    vmc.run(n_iter=40)
    duration = time.time() - start

    # VMC statistics
    E_vmc = float(vmc.energy.mean.real)
    Var_vmc = float(vmc.energy.variance.real)

    #### exact diagonalization comparison ####

    E_exact, psi_exact = calculate_groundstate(ha)

    # reconstruct variational state on full Hilbert space (2**(L*L) = 65536 for L=4)
    psi_vmc = np.asarray(vs.to_array(), dtype=np.complex128)  # normalized by default

    # compare wavefunctions
    overlap = np.vdot(psi_exact, psi_vmc)
    fidelity = float(np.abs(overlap) ** 2)
    rel_err = abs(E_vmc - E_exact) / abs(E_exact)

    print(f"Runtime (VMC): {duration:.1f}s")
    print(f"VMC Energy: {E_vmc:.8f}")
    print(f"VMC Variance: {Var_vmc:.3e}")
    print(f"Exact Energy: {E_exact:.8f}")
    print(f"Relative error: {rel_err:.3e}")
    print(f"Fidelity |<psi_exact|psi_vmc>|^2: {fidelity:.6f}")


if __name__ == "__main__":
    np.random.seed(123)
    main()
