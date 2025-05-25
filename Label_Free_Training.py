import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse

from FMT.FMT_Models import IPS, KNNLC, EFCN
from FMT.FMT_dataset import UnlabeledDataset, CombinedDataset, load_fmtd_data
from NIR_II_Guided_Diffusion.utils.train_utils import RandAugment1D

class MeanTeacher(nn.Module):
    def __init__(self, model, alpha=0.999):
        super(MeanTeacher, self).__init__()
        self.model = model
        self.teacher_model = nn.Sequential(*list(model.children()))
        self.alpha = alpha

    @staticmethod
    def update_ema(model, ema_model, alpha=0.999):
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)

    def forward(self, labeled_data, unlabeled_data, labels, device, ema_model, alpha=0.99, Lambda=0):
        self.model = self.model.to(device)
        labeled_data = labeled_data.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)

        # Supervised loss
        outputs = self.model(labeled_data)
        loss_supervised = F.mse_loss(outputs, labels)

        # Consistency loss
        unlabeled_data = unlabeled_data.to(device).float()
        ema_model.train()
        teacher_outputs = ema_model(unlabeled_data)
        student_outputs = self.model(unlabeled_data)
        loss_unsupervised = F.mse_loss(student_outputs, teacher_outputs) * Lambda

        # Update EMA model
        self.update_ema(self.model, ema_model, alpha)

        total_loss = loss_supervised + loss_unsupervised
        return total_loss

class FixMatch(nn.Module):
    def __init__(self, model, confidence=0.95):
        super().__init__()
        self.model = model
        self.confidence = confidence
        self.strong_augment = RandAugment1D(batch=8, n=3, m9=0.5, mstd=0.2)

    def forward(self, labeled_data, unlabeled_data, labels, device):
        self.model = self.model.to(device)
        
        # Supervised loss
        outputs = self.model(labeled_data)
        loss_supervised = F.mse_loss(outputs, labels)

        # Unsupervised loss with weak and strong augmentation
        weak_aug = torch.flip(unlabeled_data, dims=[1])
        strong_aug = self.strong_augment(unlabeled_data)
        
        with torch.no_grad():
            pseudo_logits = self.model(weak_aug)
        pseudo_labels = (pseudo_logits > self.confidence).float()
        
        logits_strong = self.model(strong_aug)
        loss_unsupervised = F.mse_loss(logits_strong, pseudo_labels)
        
        # Dynamic weight scaling
        scale = (loss_supervised / (loss_unsupervised + 1e-2)).float()
        total_loss = loss_supervised + loss_unsupervised * scale * 1e-2
        
        return total_loss



def train_label_free(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    train_ds, eval_ds = load_fmtd_data("DataSets", args.eval_type)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Load unlabeled data
    unlabeled_data = np.load(args.unlabeled_path).transpose((1,0))
    unlabeled_dataset = UnlabeledDataset(unlabeled_data)
    combined_dataset = CombinedDataset(train_ds, unlabeled_dataset)
    combined_loader = DataLoader(
        combined_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize models
    model = get_model(args).to(device)
    if args.ssl_method == "MT":
        ssl_model = MeanTeacher(model)
        ema_model = get_model(args).to(device)
    else:
        ssl_model = FixMatch(model)
        ema_model = None

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for labeled_data, unlabeled_data, labels in combined_loader:
            labeled_data = labeled_data.to(device)
            unlabeled_data = unlabeled_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if args.ssl_method == "MT":
                loss = ssl_model(labeled_data, unlabeled_data, labels, device, ema_model)
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
            save_path = f"{args.model_type}_{args.ssl_method}.pth"
            torch.save(model.state_dict(), save_path)

def get_model(args):
    if args.model_type == "IPS":
        return IPS()
    elif args.model_type == "EFCN":
        return EFCN()
    elif args.model_type == "KNNLC":
        return KNNLC()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssl_method', type=str, default='MT', choices=['MT', 'FixMatch'])
    parser.add_argument('--model_type', type=str, default='IPS')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--unlabeled_path', type=str, default='DataSets/guided_input.npy')
    parser.add_argument('--eval_type', type=str, default='validation')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_label_free(args)