import torch

# Device & dtypes (shared)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # real-valued RBM parameters and energies