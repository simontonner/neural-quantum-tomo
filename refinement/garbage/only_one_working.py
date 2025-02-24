import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

################################################################################
# 🔧 1) Hyperparameters & Device Setup
################################################################################

batch_size = 128
visible_units = 28 * 28  # MNIST images (flattened)
hidden_units = 256      # More expressive capacity
k = 1                    # Gibbs steps for PCD
lr = 1e-3                # Learning rate
num_epochs = 20
weight_decay = 1e-5      # L2 regularization
lr_decay = 0.95          # Learning rate decay per epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 📥 2) Data Setup: Binarized MNIST
################################################################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),               # Normalize to [0,1]
    transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

################################################################################
# 🔥 3) Define a Persistent Contrastive Divergence RBM
################################################################################

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Parameters: Weight matrix, visible bias, hidden bias
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))  # Visible bias
        self.c = nn.Parameter(torch.zeros(n_hidden))   # Hidden bias

    def sample_h_given_v(self, v):
        """Given visible v, sample hidden units h."""
        prob_h = torch.sigmoid(torch.matmul(v, self.W) + self.c)
        return torch.bernoulli(prob_h), prob_h  # Binary & prob

    def sample_v_given_h(self, h):
        """Given hidden h, sample visible units v."""
        prob_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b)
        return torch.bernoulli(prob_v), prob_v  # Binary & prob

    def free_energy(self, v):
        """Compute Free Energy of the given visible vector."""
        vbias_term = torch.matmul(v, self.b)
        wx_b = torch.matmul(v, self.W) + self.c
        hidden_term = nn.functional.softplus(wx_b).sum(dim=1)
        return -(vbias_term + hidden_term)

################################################################################
# 📊 4) Training with Persistent Contrastive Divergence
################################################################################

def train_rbm(rbm, train_loader, num_epochs, k, optimizer, scheduler=None):
    """Train the RBM using PCD."""

    # Initialize persistent fantasy particles (start with noise)
    persistent_v = torch.bernoulli(torch.rand(batch_size, rbm.n_visible)).to(device)

    rbm.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, rbm.n_visible).to(device)

            # Positive phase: Sample hidden layer given real data
            h0, _ = rbm.sample_h_given_v(data)

            # Negative phase: Gibbs Sampling (PCD)
            v_k = persistent_v  # Start from last batch's fantasy
            for _ in range(k):
                h_k, _ = rbm.sample_h_given_v(v_k)
                v_k, _ = rbm.sample_v_given_h(h_k)

            # Update the persistent fantasy particles
            persistent_v = v_k.detach()

            # Compute contrastive divergence loss
            loss = rbm.free_energy(data).mean() - rbm.free_energy(v_k).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Free Energy Loss: {total_loss / len(train_loader):.4f}")

################################################################################
# 🚀 5) Training & Optimization
################################################################################

rbm = RBM(visible_units, hidden_units).to(device)

# Adam optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)

# Learning rate decay scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

train_rbm(rbm, train_loader, num_epochs=num_epochs, k=k, optimizer=optimizer, scheduler=scheduler)

################################################################################
# 🎨 6) Generating New Images from Pure Noise
################################################################################

def sample_from_rbm(rbm, n_samples=16, n_steps=100):
    """Start from noise and run Gibbs sampling to generate new samples."""
    rbm.eval()
    v = torch.bernoulli(torch.rand(n_samples, rbm.n_visible)).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            h, _ = rbm.sample_h_given_v(v)
            v, _ = rbm.sample_v_given_h(h)
    return v.cpu().view(-1, 1, 28, 28)  # Reshape for plotting

# Generate and visualize fantasy digits
fantasy_images = sample_from_rbm(rbm, n_samples=16, n_steps=200)

# Plot fantasy images
def plot_images(images, title="Fantasy Particles"):
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

plot_images(fantasy_images, title="RBM Sampled Digits")
