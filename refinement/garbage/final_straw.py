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

# If you want a learning rate scheduler, define something like:
lr_gamma = 0.95  # multiplicative LR decay per epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 2) Data Setup: Binarized MNIST
################################################################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),               # roughly scale to [0,1]
    transforms.Lambda(lambda x: (x > 0.5).float())  # binarize
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True)

################################################################################
# 3) Define a Bernoulli-Bernoulli RBM with PCD
################################################################################

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Parameters: W is the weight matrix, b is visible bias, c is hidden bias
        # We use a uniform initialization, but you can also do Xavier/Glorot.
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.c = nn.Parameter(torch.zeros(n_hidden))   # hidden bias

    def sample_h_given_v(self, v):
        """
        Given visible v, compute P(h=1|v) and then sample a hidden layer h.
        v shape: [batch_size, n_visible]
        returns: h, prob_h. Both shape: [batch_size, n_hidden]
        """
        prob_h = torch.sigmoid(torch.matmul(v, self.W) + self.c)
        h = torch.bernoulli(prob_h)
        return h, prob_h

    def sample_v_given_h(self, h):
        """
        Given hidden h, compute P(v=1|h) and then sample a visible layer v.
        h shape: [batch_size, n_hidden]
        returns: v, prob_v. Both shape: [batch_size, n_visible]
        """
        prob_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b)
        v = torch.bernoulli(prob_v)
        return v, prob_v

    def free_energy(self, v):
        """
        Free energy F(v) = -b.T * v - sum_{j} softplus(c_j + (W^T v)_j)
        shape of v: [batch_size, n_visible]
        returns: 1-D tensor of size [batch_size]
        """
        vbias_term = torch.matmul(v, self.b)
        wx_b = torch.matmul(v, self.W) + self.c  # shape: [batch_size, n_hidden]
        hidden_term = nn.functional.softplus(wx_b).sum(dim=1)
        return -(vbias_term + hidden_term)

################################################################################
# 4) Training with Persistent Contrastive Divergence
################################################################################

def train_rbm(rbm, train_loader, num_epochs, k, optimizer, scheduler=None):
    # We keep one persistent chain per training example in the batch
    # or keep it smaller if you want. Here let's just store one chain per batch.
    # We'll re-init each epoch for simplicity or you can keep it persistent across epochs.

    rbm.train()
    for epoch in range(num_epochs):
        # Re-init the persistent chain (size = batch_size)
        # This means we start from random noise. One can keep it from last epoch.
        persistent_v = torch.bernoulli(torch.rand(batch_size, rbm.n_visible)).to(device)

        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, rbm.n_visible).to(device)

            # Positive phase
            # h_0 sample from p(h|v0)
            h0, _ = rbm.sample_h_given_v(data)

            # Negative phase (PCD-k)
            v_k = persistent_v  # start from last chain
            for step in range(k):
                h_k, _ = rbm.sample_h_given_v(v_k)
                v_k, _ = rbm.sample_v_given_h(h_k)

            # Now v_k is the "fantasy particles"
            # We update the persistent chain
            persistent_v = v_k.detach()

            # h_k sample from p(h|v_k)
            h_k, _ = rbm.sample_h_given_v(v_k)

            # Compute the free-energy difference (loss) for CD
            # F(v0) - F(v_k) is what we want to minimize
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

# We'll use Adam with weight decay. This is a rough stand-in for L2.
optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)

# Optionally define a LR scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

train_rbm(rbm, train_loader, num_epochs=num_epochs, k=k, optimizer=optimizer, scheduler=scheduler)

################################################################################
# 6) Generating new images from random noise
################################################################################

def sample_from_rbm(rbm, n_samples=16, n_steps=100):
    """
    Start from random noise (pure uniform Bernoulli). Then do n_steps of Gibbs sampling.
    Return the final visible states for visualization.
    """
    rbm.eval()
    v = torch.bernoulli(torch.rand(n_samples, rbm.n_visible)).to(device)
    with torch.no_grad():
        for step in range(n_steps):
            h, _ = rbm.sample_h_given_v(v)
            v, _ = rbm.sample_v_given_h(h)
    return v.cpu().view(-1, 1, 28, 28)  # reshape for plotting

# Let's generate a small grid of fantasy digits
fantasy = sample_from_rbm(rbm, n_samples=4, n_steps=200)

# Quick plot
def plot_images(images, title="Fantasy Particles"):
    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

plot_images(fantasy, title="RBM Sampled Digits")
