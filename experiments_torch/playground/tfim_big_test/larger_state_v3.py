import os
import time
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"         # cpu runs faster. at least on my machine
os.environ["JAX_ENABLE_X64"] = "0"          # use 64-bit floats for higher precision

import jax
import netket as nk

def main():
    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())
    print("NetKet version:", nk.__version__)

    L = 4
    g = nk.graph.Grid(extent=[L, L], pbc=True)

    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # --- symmetries & GCNN ansatz ---
    # Use all graph automorphisms to build a symmetry-equivariant CNN-like ansatz.
    # This is stronger than a plain RBM on 2D.
    symms = g.automorphisms()

    # Reasonable medium-size GCNN:
    # - 4 layers
    # - 16 features per layer
    # You can bump features if training is stable.
    ma = nk.models.GCNN(symmetries=symms, layers=4, features=16, complex_output=True)


    # --- sampler & variational state ---
    # Exchange moves preserve total_sz=0.
    # Use many chains and large chunk_size to keep JAX kernels fat on CPU.
    sa = nk.sampler.MetropolisExchange(hilbert=hi,graph=g,
        n_chains=512,
        sweep_size=2,
    )

    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=8192,
        chunk_size=8192,
        n_discard_per_chain=64,
        seed=123,
    )

    #### optimize with ADAM and natural gradient (SR) ####

    opt = nk.optimizer.Adam(learning_rate=0.005)
    sr = nk.optimizer.SR(diag_shift=0.01)     # let NetKet choose its default linear solver (CG-like)

    vmc = nk.driver.VMC(hamiltonian=ha, optimizer=opt, variational_state=vs, preconditioner=sr)

    # --- run ---
    t0 = time.time()
    vmc.run(n_iter=400, out="vmc_6x6_gcnn")  # logs/checkpoints to folder
    dur = time.time() - t0

    E = float(vs.energy.mean.real)
    Var = float(vs.energy.variance.real)

    print(f"Runtime: {dur:.1f}s")
    print(f"Energy: {E:.8f}")
    print(f"Variance: {Var:.3e}")

if __name__ == "__main__":
    np.random.seed(123)
    main()
