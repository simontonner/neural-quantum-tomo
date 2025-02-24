import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

################################################################################
# 1) Hyperparameters
################################################################################

batch_size = 128
visible_units = 28 * 28  # MNIST
hidden_units = 256       # Try varying this
k = 1                    # Number of Gibbs sampling steps in PCD
lr = 1e-3                # Initial learning rate
weight_decay = 1e-5      # L2 penalty on weights
num_epochs = 10          # You might want more for better results

# Optionally, use a learning rate scheduler
lr_gamma = 0.95  # multiplicative LR decay per epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 2) Data Setup: Binarized MNIST
################################################################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),               # roughly scale to [0,1]
    transforms.Lambda(lambda x: (x > 0.5).float())    # binarize
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)

################################################################################
# 3) Define a Bernoulli-Bernoulli RBM with PCD (Renamed Methods)
################################################################################

class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # Parameters: W is the weight matrix, a is visible bias, b is hidden bias.
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.a = nn.Parameter(torch.zeros(visible_dim))  # visible bias
        self.b = nn.Parameter(torch.zeros(hidden_dim))   # hidden bias

    def sample_hidden(self, v):
        """
        Given visible v, compute P(h=1|v) and sample h.
        v shape: [batch_size, visible_dim]
        returns: h_sample, h_probs (both shape: [batch_size, hidden_dim])
        """
        h_probs = torch.sigmoid(torch.matmul(v, self.W) + self.b)
        h_sample = torch.bernoulli(h_probs)
        return h_sample, h_probs

    def sample_visible(self, h):
        """
        Given hidden h, compute P(v=1|h) and sample v.
        h shape: [batch_size, hidden_dim]
        returns: v_sample, v_probs (both shape: [batch_size, visible_dim])
        """
        v_probs = torch.sigmoid(torch.matmul(h, self.W.t()) + self.a)
        v_sample = torch.bernoulli(v_probs)
        return v_sample, v_probs

    def free_energy(self, v):
        """
        Free energy F(v) = -a^T v - sum_j softplus(b_j + (W^T v)_j)
        v shape: [batch_size, visible_dim]
        returns: 1-D tensor [batch_size]
        """
        vbias_term = torch.matmul(v, self.a)
        wx_b = torch.matmul(v, self.W) + self.b
        hidden_term = nn.functional.softplus(wx_b).sum(dim=1)
        return -(vbias_term + hidden_term)

################################################################################
# 4) Training with Persistent Contrastive Divergence
################################################################################

def train_rbm(rbm, train_loader, num_epochs, k, optimizer, scheduler=None):
    rbm.train()
    for epoch in range(num_epochs):
        # Re-init persistent chain (size = batch_size)
        persistent_v = torch.bernoulli(torch.rand(batch_size, rbm.visible_dim)).to(device)

        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, rbm.visible_dim).to(device)

            # Positive phase
            h0, _ = rbm.sample_hidden(data)

            # Negative phase (PCD-k)
            v_k = persistent_v  # start from last chain
            for _ in range(k):
                h_k, _ = rbm.sample_hidden(v_k)
                v_k, _ = rbm.sample_visible(h_k)

            # Update persistent chain
            persistent_v = v_k.detach()

            # Compute the free-energy difference (loss)
            loss = rbm.free_energy(data).mean() - rbm.free_energy(v_k).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss (Free Energy diff): {avg_loss:.4f}")

################################################################################
# 5) Putting it all together: Instantiate and train
################################################################################

rbm = RBM(visible_units, hidden_units).to(device)

optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

train_rbm(rbm, train_loader, num_epochs=num_epochs, k=k, optimizer=optimizer, scheduler=scheduler)

################################################################################
# 6) Generating new images from random noise
################################################################################

def sample_from_rbm(rbm, n_samples=16, n_steps=100):
    """
    Start from random noise and perform n_steps of Gibbs sampling.
    Returns final visible states for visualization.
    """
    rbm.eval()
    v = torch.bernoulli(torch.rand(n_samples, rbm.visible_dim)).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            h, _ = rbm.sample_hidden(v)
            v, _ = rbm.sample_visible(h)
    return v.cpu().view(-1, 1, 28, 28)  # reshape for plotting

# Generate a small grid of fantasy digits
fantasy = sample_from_rbm(rbm, n_samples=4, n_steps=200)

# Quick plot function
def plot_images(images, title="Fantasy Particles"):
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

plot_images(fantasy, title="RBM Sampled Digits")
