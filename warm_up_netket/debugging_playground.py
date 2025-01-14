import netket as nk


L = 4
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
sx_1 = nk.operator.spin.sigmax(hi, 1)
print(sx_1)
