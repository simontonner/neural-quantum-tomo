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



def train_rbm(rbm, train_loader, num_epochs, k, optimizer, scheduler=None, pcd_reset=5):
    rbm.train()

    fantasy_particles = torch.bernoulli(torch.rand(batch_size, rbm.n_visible)).to(device)

    metrics = {}
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, rbm.n_visible).to(device)

            if batch_idx % pcd_reset == 0:
                fantasy_particles = torch.bernoulli(torch.rand(batch_size, rbm.n_visible)).to(device)

            v_k = rbm.sample_gibbs(fantasy_particles, k)
            fantasy_particles = v_k.detach()

            loss = rbm.free_energy(data).mean() - rbm.free_energy(v_k).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        metrics[epoch] = { "free_energy_loss": avg_loss }
        print(f"Epoch [{epoch+1}/{num_epochs}] - Free Energy Loss: {avg_loss:.4f}")

    return metrics


#### TRAINING

batch_size      = 128
visible_units   = 28*28
hidden_units    = 256
k               = 1
lr              = 1e-3
num_epochs      = 30
pcd_reset       = 75        # reset persistent chain every N batches
weight_decay    = 1e-5      # L2 regularization
lr_decay        = 0.95      # learning rate decay PER EPOCH


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

rbm = RBM(visible_units, hidden_units).to(device)

optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

metrics = train_rbm(rbm, train_loader, num_epochs=num_epochs, k=k, optimizer=optimizer, scheduler=scheduler, pcd_reset=pcd_reset)