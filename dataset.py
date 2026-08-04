import torch
from torch.utils.data import Dataset
import scipy.io
import os

class NSVorticityDataset(Dataset):
    def __init__(self, mat_file_path, split="train", t_in=10, t_out=10):
        """
        Loads the Navier-Stokes vorticity dataset.
        Assumes legacy .mat format (v7.3 or earlier).
        Shape of 'u': (1200, 64, 64, 20)
        """
        self.t_in = t_in
        self.t_out = t_out
        
        # In a real scenario, this would load the .mat file.
        # For syntactic completeness and testing if the file doesn't exist, we will fallback to random data
        if os.path.exists(mat_file_path):
            data = scipy.io.loadmat(mat_file_path)
            # data['u'] shape: (1200, 64, 64, 20)
            u = torch.tensor(data['u'], dtype=torch.float32)
        else:
            print(f"Warning: Dataset {mat_file_path} not found. Using random data for structural testing.")
            u = torch.randn(1200, 64, 64, 20, dtype=torch.float32)
            
        if split == "train":
            # first 1000 trajectories
            self.u = u[:1000]
        elif split == "test":
            # last 200 trajectories
            self.u = u[1000:1200]
        else:
            raise ValueError("split must be 'train' or 'test'")
            
    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        # Input: first t_in timesteps
        # Target: next t_out timesteps
        x = self.u[idx, ..., :self.t_in] # (64, 64, 10)
        y = self.u[idx, ..., self.t_in:self.t_in+self.t_out] # (64, 64, 10)
        
        # PyTorch expects channel first: (C, H, W)
        x = x.permute(2, 0, 1) # (10, 64, 64)
        y = y.permute(2, 0, 1) # (10, 64, 64)
        
        return x, y
