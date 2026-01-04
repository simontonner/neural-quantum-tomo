# NetKet VMC on a 6x6 Heisenberg lattice with SR (natural gradient)
import time
import netket as nk
import numpy as np

def main():
    print("NetKet version:", nk.__version__)
    L = 6
    pbc = False  # set True if you want periodic boundaries

    # Lattice and Hilbert (N=36 is even, so total_sz=0 is allowed)
    g = nk.graph.Grid(extent=[L, L], pbc=pbc)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=0)

    # Heisenberg AFM
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Ansatz, sampler, variational state
    ma = nk.models.RBM(alpha=2)  # bump alpha if noisy (e.g., 4)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, n_chains=64)
    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=4096,
        n_discard_per_chain=64,
        seed=123,
    )

    # Optimizer + SR preconditioner
    opt = nk.optimizer.Sgd(learning_rate=0.02)
    sr = nk.optimizer.SR(diag_shift=0.01)

    # VMC driver with SR
    vmc = nk.driver.VMC(hamiltonian=ha, optimizer=opt, variational_state=vs, preconditioner=sr)

    t0 = time.time()
    vmc.run(n_iter=1000, out="vmc_6x6")  # logs and checkpoints in vmc_6x6/
    dur = time.time() - t0

    E = float(vs.energy.mean.real)
    Var = float(vs.energy.variance.real)
    print(f"Runtime: {dur:.1f}s | E: {E:.8f} | Var: {Var:.3e}")

if __name__ == "__main__":
    np.random.seed(123)
    main()
