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

    #### define the hamiltonian ####

    L = 4
    g = nk.graph.Grid(extent=[L, L], pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    #### use a GCNN ansatz with symmetries ####

    symms = g.automorphisms()
    ma = nk.models.GCNN(symmetries=symms, layers=4, features=16, complex_output=True)

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

    #### optimize with ADAM and natural gradient (SR) ####

    opt = nk.optimizer.Adam(learning_rate=0.005)
    sr = nk.optimizer.SR(diag_shift=0.01)     # let NetKet choose its default linear solver (CG-like)
    vmc = nk.driver.VMC(hamiltonian=ha, optimizer=opt, variational_state=vs, preconditioner=sr)


    start = time.time()
    vmc.run(n_iter=40)
    duration = time.time() - start

    E = float(vs.energy.mean.real)
    Var = float(vs.energy.variance.real)

    print(f"Runtime: {duration:.1f}s")
    print(f"Energy: {E:.8f}")
    print(f"Variance: {Var:.3e}")

if __name__ == "__main__":
    np.random.seed(123)
    main()
