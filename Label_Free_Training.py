import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from FMT.FMT_Models import IPS, KNNLC, EFCN
from FMT.FMT_dataset import UnlabeledDataset, CombinedDataset, load_fmtd_data
from FMT.SSL_Algorithms import MeanTeacher, FixMatch, MCNet
from config import get_config

def get_model(config):
    """Get base model according to model type"""
    if config.ssl_model_type == "IPS":
        return IPS()
    elif config.ssl_model_type == "EFCN":
        return EFCN()
    elif config.ssl_model_type == "KNNLC":
        return KNNLC()
    else:
        raise ValueError(f"Unknown model type: {config.ssl_model_type}")

def get_ssl_model(config, base_model, device):
    """Get SSL model based on method type"""
    if config.ssl_method == "MT":
        ssl_model = MeanTeacher(
            base_model,
            alpha=config.mt_ema_alpha
        )
        ema_model = get_model(config).to(device)
        return ssl_model, ema_model
    
    elif config.ssl_method == "FixMatch":
        ssl_model = FixMatch(
            base_model,
            confidence=config.fixmatch_confidence
        )
        return ssl_model, None
    
    elif config.ssl_method == "MCNet":
        ssl_model = MCNet(
            base_model,
            dropout_rate=config.mcnet_dropout_rate,
            n_decoders=config.mcnet_n_decoders
        )
        return ssl_model, None
    
    raise ValueError(f"Unknown SSL method: {config.ssl_method}")

def train_label_free(config):
    """Main training function using config parameters"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    train_ds, eval_ds = load_fmtd_data("DataSets", config.ssl_eval_type)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.ssl_batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Load unlabeled data
    unlabeled_data = np.load(config.unlabeled_path).transpose((1,0))
    unlabeled_dataset = UnlabeledDataset(unlabeled_data)
    combined_dataset = CombinedDataset(train_ds, unlabeled_dataset)
    combined_loader = DataLoader(
        combined_dataset, 
        batch_size=config.ssl_batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize models
    base_model = get_model(config).to(device)
    ssl_model, ema_model = get_ssl_model(config, base_model, device)

    # Setup training
    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=config.ssl_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.ssl_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float('inf')

    # Training loop
    for epoch in range(config.ssl_epochs):
        base_model.train()
        total_loss = 0
        
        for labeled_data, unlabeled_data, labels in combined_loader:
            labeled_data = labeled_data.to(device)
            unlabeled_data = unlabeled_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with method-specific parameters
            if config.ssl_method == "MT":
                loss = ssl_model(
                    labeled_data, 
                    unlabeled_data, 
                    labels, 
                    device, 
                    ema_model,
                    alpha=config.mt_ema_alpha,
                    Lambda=config.mt_consistency_weight
                )
            else:
                loss = ssl_model(labeled_data, unlabeled_data, labels, device)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(combined_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}')
        scheduler.step()

        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
            save_path = Path(f"checkpoints/{config.ssl_model_type}_{config.ssl_method}.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(base_model.state_dict(), save_path)

if __name__ == "__main__":
    config = get_config()
    train_label_free(config)