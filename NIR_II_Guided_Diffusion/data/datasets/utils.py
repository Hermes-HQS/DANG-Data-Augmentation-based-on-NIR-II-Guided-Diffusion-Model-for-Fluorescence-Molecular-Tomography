import numpy as np
import torch
from torch.utils.data import Dataset

class MatDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: Data loaded from .mat file (numpy array)
            transform: Optional image transform
        """
        # Extract data from the dictionary returned by loadmat
        if isinstance(data, dict):
            self.data = data['__your__data_key__']  # Replace with the actual key in your .mat file
        else:
            self.data = data
            
        # Convert dimension order to PyTorch format (batch, channel, height, width)
        # If your images are grayscale, you need to add a channel dimension
        self.data = np.transpose(self.data, (2, 0, 1))  # From (H,W,N) to (N,H,W)
        self.data = self.data[:, np.newaxis, :, :]  # Add channel dimension (N,1,H,W)
        
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to torch.Tensor
        sample = torch.from_numpy(sample).float().requires_grad_(True)
        
        if self.transform:
            sample = self.transform(sample)
            
        # Return image tensor (1,H,W)
        return sample