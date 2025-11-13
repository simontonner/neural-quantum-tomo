# NetKet VMC (JAX on CPU) for a 6x6 Heisenberg model, vectorized for speed
import os, time
os.environ["JAX_PLATFORMS"] = "cpu"   # force JAX to CPU
os.environ["JAX_ENABLE_X64"] = "1"    # higher numerical precision

import jax
import netket as nk
import numpy as np

def main():
    print("JAX backend:", jax.default_backend())
    print("Devices:", jax.devices())

    L, pbc = 6, False
    g  = nk.graph.Grid(extent=[L, L], pbc=pbc)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=0)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Lightweight ansatz; bump alpha if needed
    ma = nk.models.RBM(alpha=2)

    # Heavily vectorized sampling (CPU-friendly)
    sa = nk.sampler.MetropolisExchange(
        hilbert=hi, graph=g, n_chains=1024, n_sweeps=1
    )
    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=8192,            # keep big to amortize JAX compute
        chunk_size=8192,
        n_discard_per_chain=64,
        seed=123,
    )

    # Optimizer + SR (natural gradient via CG)
    opt = nk.optimizer.Adam(learning_rate=0.01)
    sr  = nk.optimizer.SR(diag_shift=0.05, solver="cg")

    vmc = nk.driver.VMC(hamiltonian=ha, optimizer=opt, variational_state=vs, preconditioner=sr)

    t0 = time.time()
    vmc.run(n_iter=600, out="vmc_6x6_cpujax")   # logs/checkpoints in this folder
    dur = time.time() - t0

    E   = float(vs.energy.mean.real)
    Var = float(vs.energy.variance.real)
    print(f"Runtime: {dur:.1f}s | E: {E:.8f} | Var: {Var:.3e}")

if __name__ == "__main__":
    np.random.seed(123)
    main()
