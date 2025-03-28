import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import matplotlib.pyplot as plt

import os
import gzip
import numpy as np
import requests
from tqdm import tqdm

data_dir = "./data"
device = jax.devices('cpu')[0]

print(f"Data resides in        : {data_dir}")
print(f"Training model on      : {str(device)}")


class MNISTDataset:
    def __init__(self, root="./data", train=True, download=True, transform=None):
        self.root = os.path.join(root, "MNIST", "raw")
        self.train = train
        self.transform = transform

        self.download_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

        self.files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }

        if download:
            self._download_if_needed()

        img_file = self.files["train_images" if train else "test_images"]
        lbl_file = self.files["train_labels" if train else "test_labels"]

        self.images = self._load_images(os.path.join(self.root, img_file))
        self.labels = self._load_labels(os.path.join(self.root, lbl_file))

    def _download_if_needed(self):
        os.makedirs(self.root, exist_ok=True)
        for filename in self.files.values():
            path = os.path.join(self.root, filename)
            if not os.path.exists(path):
                print(f"Downloading {filename}...")
                self._download_file(self.download_url + filename, path)

    def _download_file(self, url, path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(path, 'wb') as f, tqdm(
                    desc=path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")

    def _load_images(self, path):
        with gzip.open(path, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28)

    def _load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        batch = []
        for idx in self.indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                images, labels = zip(*batch)
                yield np.stack(images), np.array(labels)
                batch = []

        if batch and not self.drop_last:
            images, labels = zip(*batch)
            yield np.stack(images), np.array(labels)

    def __len__(self):
        return len(self.indices)


def print_samples(samples, elements_per_row=10, fig_width=10, cmap="binary"):
    num_digits = len(samples)
    num_rows = (num_digits + elements_per_row - 1) // elements_per_row

    plt.figure(figsize=(fig_width, fig_width / elements_per_row * num_rows))
    for idx, (label, image) in enumerate(samples):
        plt.subplot(num_rows, elements_per_row, idx + 1)
        plt.imshow(image.squeeze(), cmap=cmap)
        plt.title(label, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


transform = lambda x: ((x / 255.0) > 0.5).astype(jnp.float32)

train_dataset = MNISTDataset(root=data_dir, train=True, download=True, transform=transform)

sample_list = [(label, next(image for image, lbl in train_dataset if lbl == label)) for label in range(10)]
print_samples(sample_list)


class RBM(nn.Module):
    n_visible: int
    n_hidden: int

    def setup(self):
        self.W = self.param("W", nn.initializers.normal(0.01), (self.n_visible, self.n_hidden))
        self.b = self.param("b", nn.initializers.zeros, (self.n_visible,))
        self.c = self.param("c", nn.initializers.zeros, (self.n_hidden,))

    def _sample_hidden(self, v, T=1.0):
        key = self.make_rng("sample")
        logits = (v @ self.W + self.c) / T
        h_probs = jax.nn.sigmoid(logits)
        h_sample = jax.random.bernoulli(key, h_probs)
        return h_sample, h_probs

    def _sample_visible(self, h, T=1.0):
        key = self.make_rng("sample")
        logits = (h @ self.W.T + self.b) / T
        v_probs = jax.nn.sigmoid(logits)
        v_sample = jax.random.bernoulli(key, v_probs)
        return v_sample, v_probs

    def sample_gibbs(self, v0_sample, k=1, T=1.0):
        v = v0_sample
        for _ in range(k):
            h, _ = self._sample_hidden(v, T)
            v, _ = self._sample_visible(h, T)
        return v

    def free_energy(self, v):
        visible_term = jnp.dot(v, self.b)
        hidden_term = jnp.sum(jax.nn.softplus(v @ self.W + self.c), axis=1)
        return -visible_term - hidden_term

    def generate(self, params, n_samples=16, T_schedule=None, seed=0):
        key = jax.random.PRNGKey(seed)
        rbm = self.bind({"params": params}, rngs={"sample": key})
        v = jax.random.bernoulli(key, shape=(n_samples, self.n_visible)).astype(jnp.float32)

        for i, T in enumerate(T_schedule):
            key = jax.random.fold_in(key, i)
            rbm = rbm.replace_rngs({"sample": key})
            v = rbm.sample_gibbs(v, k=1, T=T)

        return v

    def __call__(self, v):
        return v # flax linen requires a __call__ method


OK CHATGPT. SO FAR  I HAVE EVERYTHING NICELY TRANSLATED INTO JAX, PLEASE HELP ME TO TRANSLATE THIS TRAINING ROUTINE TOO:

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
num_epochs      = 40
pcd_reset       = 75        # reset persistent chain every N batches
weight_decay    = 1e-5      # L2 regularization
lr_decay        = 0.95      # learning rate decay PER EPOCH


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

rbm = RBM(visible_units, hidden_units).to(device)

optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

metrics = train_rbm(rbm, train_loader, num_epochs=num_epochs, k=k, optimizer=optimizer, scheduler=scheduler, pcd_reset=pcd_reset)