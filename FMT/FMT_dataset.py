import numpy as np
import torch
from scipy import sparse
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

def load_fmt_image(path):
    """
    Load FMT image data from specified path
    Args:
        path: Path to the data directory
    Returns:
        train_ds: Training dataset
        eval_ds: Evaluation dataset
    """
    train_data_path = '__your__dataset__path'  # Replace with your dataset path
    train_data = np.load(train_data_path)

    train_labels = np.zeros((1965, 1))
    train_data_transposed = np.transpose(train_data, (2, 0, 1))

    train_tensor = torch.tensor(train_data_transposed, dtype=torch.float32)
    label_tensor = torch.tensor(train_labels, dtype=torch.float32)
    train_tensor = train_tensor.unsqueeze(1)
    train_ds = TensorDataset(train_tensor, label_tensor)
    eval_ds = train_ds

    return train_ds, eval_ds


def load_fmtd_data(path, test_type='single'):
    """
    Load FMT training and evaluation data
    Args:
        path: Path to the data directory
        test_type: Type of test data to load ('single', 'big_assemble', 'double', 'three', 'four')
    Returns:
        train_ds: Training dataset
        eval_ds: Evaluation dataset
    """
    # Load training data
    train_data_path = '__your__training__dataset__input__path__'  # Replace with your dataset path
    train_label_path = '__your__training__dataset__label__path__'  # Replace with your dataset path
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)

    # Load evaluation data
    eval_data_path = '__your__evaluation__dataset__input__path__'  # Replace with your dataset path
    eval_label_path = '__your__evaluation__dataset__label__path__'  # Replace with your dataset path
    eval_data = np.load(eval_data_path)
    eval_label = np.load(eval_label_path)

    # Index list for data selection
    idx_list = list(range(1965 * 4)) + list(range(1965 * 6, 25790))

    # Convert to PyTorch tensors
    train_input = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)
    eval_input = torch.tensor(eval_data, dtype=torch.float32)
    eval_label = torch.tensor(eval_label, dtype=torch.float32)

    # Scale evaluation input
    eval_input_scaled = eval_input

    # Transpose dimensions for proper format
    train_input_transposed = train_input.transpose(0, 1)
    train_label_transposed = train_label.transpose(0, 1)
    eval_input_transposed = eval_input_scaled.transpose(0, 1)
    eval_label_transposed = eval_label.transpose(0, 1)

    # Create TensorDatasets
    train_ds = TensorDataset(train_input_transposed, train_label_transposed)
    eval_ds = TensorDataset(eval_input_transposed, eval_label_transposed)

    return train_ds, eval_ds


class CombinedDataset(Dataset):
    """
    Dataset class that combines labeled and unlabeled data
    Args:
        labeled_dataset: Dataset containing labeled samples
        unlabeled_dataset: Dataset containing unlabeled samples
    """
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return len(self.labeled_dataset)

    def __getitem__(self, idx):
        labeled_data, labeled_label = self.labeled_dataset[idx]
        # Use modulo to cycle through unlabeled dataset
        unlabeled_idx = idx % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]

        return labeled_data, unlabeled_data, labeled_label


class UnlabeledDataset(Dataset):
    """
    Dataset class for unlabeled data
    Args:
        data: Unlabeled data samples
        transform: Optional transform to be applied on the data
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
            sample = sample.float()
        return sample