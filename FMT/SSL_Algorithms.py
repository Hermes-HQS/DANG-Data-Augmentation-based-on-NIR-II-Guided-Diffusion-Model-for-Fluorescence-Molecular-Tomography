import torch
import torch.nn as nn
import torch.nn.functional as F
from NIR_II_Guided_Diffusion.utils.train_utils import RandAugment1D

class MeanTeacher(nn.Module):
    """Mean Teacher algorithm for semi-supervised learning"""
    def __init__(self, model, alpha=0.999):
        super(MeanTeacher, self).__init__()
        self.model = model
        self.teacher_model = nn.Sequential(*list(model.children()))
        self.alpha = alpha

    @staticmethod
    def update_ema(model, ema_model, alpha=0.999):
        """Update exponential moving average of model parameters"""
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)

    def forward(self, labeled_data, unlabeled_data, labels, device, ema_model, alpha=0.99, Lambda=0.1):
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
        loss_unsupervised = F.mse_loss(student_outputs, teacher_outputs)

        # Update EMA model
        self.update_ema(self.model, ema_model, alpha)

        return loss_supervised + Lambda * loss_unsupervised

class FixMatch(nn.Module):
    """FixMatch algorithm for semi-supervised learning"""
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
        return loss_supervised + loss_unsupervised * scale * 1e-2

class MCNet(nn.Module):
    """MCNet algorithm for semi-supervised learning"""
    def __init__(self, model, dropout_rate=0.2, n_decoders=3):
        super(MCNet, self).__init__()
        self.model = model
        self.n_decoders = n_decoders
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def _get_decoder_output(self, x):
        """Generate diverse decoder outputs using dropout"""
        return self.model(self.dropout(x))
    
    def forward(self, labeled_data, unlabeled_data, labels, device):
        self.model = self.model.to(device)
        
        # Supervised loss
        outputs_labeled = []
        loss_supervised = 0
        for _ in range(self.n_decoders):
            output = self._get_decoder_output(labeled_data)
            outputs_labeled.append(output)
            loss_supervised += F.mse_loss(output, labels)
        loss_supervised = loss_supervised / self.n_decoders
        
        # Consistency loss
        outputs_unlabeled = [self._get_decoder_output(unlabeled_data) 
                           for _ in range(self.n_decoders)]
        
        loss_consistency = 0
        for i in range(self.n_decoders):
            for j in range(i + 1, self.n_decoders):
                loss_consistency += F.mse_loss(
                    outputs_unlabeled[i], 
                    outputs_unlabeled[j]
                )
        loss_consistency = loss_consistency / (self.n_decoders * (self.n_decoders - 1) / 2)
        
        return loss_supervised + 0.1 * loss_consistency