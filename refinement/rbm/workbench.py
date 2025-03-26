class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.c = nn.Parameter(torch.zeros(n_hidden))

    def sample_hidden(self, v):
        h_probs = torch.sigmoid(v @ self.W + self.c)
        h_sample = torch.bernoulli(h_probs)
        return h_sample, h_probs

    def sample_visible(self, h):
        v_probs = torch.sigmoid(h @ self.W.t() + self.b)
        v_sample = torch.bernoulli(v_probs)
        return v_sample, v_probs

    def sample_gibbs(self, v0_sample, k=1):
        v = v0_sample
        for _ in range(k):
            h, _ = self.sample_hidden(v)
            v, _ = self.sample_visible(h)
        return v

    def free_energy(self, v):
        visible_term = v @ self.b
        hidden_term = nn.functional.softplus(v @ self.W + self.c).sum(dim=1)
        return -visible_term - hidden_term



