# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class MelSpectrogramInpainter:
# #     def __init__(self, hidden_channels, max_missing_width, model_type='adaptive_unet', n_mels=128, device='cuda'):
# #         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
# #         self.model_type = model_type
        
# #         if model_type == 'adaptive_unet':
# #             from models.unet import AdaptiveUNet
# #             # Model expects 2 channels (Channel 0: Before, Channel 1: After)
# #             self.model = AdaptiveUNet(n_mels=n_mels).to(self.device)
# #         else:
# #             raise ValueError(f"Unknown model type: {model_type}")
        
# #         self.reconstruction_loss = nn.MSELoss()
# #         self.spectral_loss = nn.L1Loss()
        
# #         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
# #         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
# #         self.training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
# #         self.current_epoch = 0
# #         self.best_val_loss = float('inf')

# #     def compute_loss(self, pred, target):
# #         # Ensure sizes match (target might be padded differently than prediction)
# #         if pred.shape != target.shape:
# #             # Resize prediction to match target exactly for loss calculation
# #             pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
# #         rec_loss = self.reconstruction_loss(pred, target)
# #         spec_loss = self.spectral_loss(torch.abs(pred), torch.abs(target))
# #         total_loss = rec_loss + 0.5 * spec_loss
# #         return total_loss, {'reconstruction': rec_loss, 'spectral': spec_loss}
    
# #     def train_step(self, before_part, after_part, target):
# #         self.model.train()
# #         self.optimizer.zero_grad()
        
# #         # Concatenate on Channel dimension (dim 1)
# #         # Input to model: [B, 2, n_mels, context_width]
# #         x_input = torch.cat([before_part, after_part], dim=1)
        
# #         target_width = target.shape[-1]
        
# #         # Forward pass
# #         pred = self.model(x_input, target_width=target_width)
        
# #         loss, loss_dict = self.compute_loss(pred, target)
        
# #         loss.backward()
# #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
# #         self.optimizer.step()
        
# #         return loss.item(), loss_dict, pred
    
# #     def predict(self, before_part, after_part, target_width=None):
# #         self.model.eval()
# #         with torch.no_grad():
# #             x_input = torch.cat([before_part, after_part], dim=1)
# #             pred = self.model(x_input, target_width=target_width)
# #         return pred

# #     # ... save_checkpoint and load_checkpoint remain the same ...
# #     def save_checkpoint(self, save_path, is_best=False):
# #         checkpoint = {
# #             'epoch': self.current_epoch,
# #             'model_state_dict': self.model.state_dict(),
# #             'optimizer_state_dict': self.optimizer.state_dict(),
# #             'scheduler_state_dict': self.scheduler.state_dict(),
# #             'training_history': self.training_history,
# #             'best_val_loss': self.best_val_loss,
# #             'model_type': self.model_type
# #         }
# #         torch.save(checkpoint, save_path)
# #         if is_best:
# #             best_path = str(save_path).replace('.pth', '_best.pth')
# #             torch.save(checkpoint, best_path)
    
# #     def load_checkpoint(self, checkpoint_path):
# #         """Load model checkpoint"""
# #         try:
# #             checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
# #             self.model.load_state_dict(checkpoint['model_state_dict'])
# #             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
# #             self.training_history = checkpoint.get('training_history', self.training_history)
# #             self.current_epoch = checkpoint.get('epoch', 0)
# #             self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
# #             print(f"Checkpoint loaded from {checkpoint_path}")
# #             return checkpoint
            
# #         except Exception as e:
# #             print(f"Error loading checkpoint: {e}")
# #             return None       



# # import torch
# # import torch.nn as nn
# # import os
# # from pathlib import Path

# # class MelSpectrogramInpainter:
# #     def __init__(self, hidden_channels, max_context_width, max_missing_width, model_type='fc_baseline', n_mels=128, device='cuda'):
# #         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
# #         self.model_type = model_type
# #         self.n_mels = n_mels
        
# #         # Norm Constants
# #         self.min_db = -80.0
# #         self.max_db = 0.0

# #         if model_type == 'fc_baseline':
# #             from models.fc_baseline import FCInpainter
# #             # We use a context of 64 for the baseline to keep weights manageable
# #             self.model = FCInpainter(n_mels=n_mels, context_width=max_context_width).to(self.device)
# #         elif model_type == 'adaptive_unet':
# #             from models.unet import AdaptiveUNet
# #             self.model = AdaptiveUNet(n_mels=n_mels).to(self.device)
        
# #         self.criterion = nn.MSELoss()
# #         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
# #         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

# #         self.training_history = {'train_loss': [], 'val_loss': []}
# #         self.current_epoch = 0
# #         self.best_val_loss = float('inf')

# #     def _norm(self, x): return (x - self.min_db) / (self.max_db - self.min_db)
# #     def _denorm(self, x): return x * (self.max_db - self.min_db) + self.min_db

# #     def train_step(self, before, after, target):
# #         self.model.train()
# #         self.optimizer.zero_grad()
        
# #         # Normalize
# #         before, after, target = self._norm(before), self._norm(after), self._norm(target)
# #         # Concatenate on Channel dimension (dim 1)
# #         # Input to model: [B, 2, n_mels, context_width]
# #         x_input = torch.cat([before, after], dim=1)

# #         # Forward (passing target.shape[-1] handles the variable width)
# #         #pred = self.model(before, after, target.shape[-1])
# #         target_width=target.shape[-1]
# #         pred = self.model(x_input, target_width=target_width)

# #         loss = self.criterion(pred, target)
# #         loss.backward()
# #         self.optimizer.step()
        
# #         return loss.item(), {'mse': loss.item()}, pred

# #     def validate_step(self, before, after, target):
# #         self.model.eval()
# #         with torch.no_grad():
# #             before, after, target = self._norm(before), self._norm(after), self._norm(target)
# #             pred = self.model(before, after, target.shape[-1])
# #             loss = self.criterion(pred, target)
# #         return loss.item(), pred

# #     def predict(self, before, after, target_width):
# #         self.model.eval()
# #         with torch.no_grad():
# #             pred = self.model(self._norm(before), self._norm(after), target_width)
# #             return self._denorm(pred)

# #     def save_checkpoint(self, path, is_best=False):
# #         state = {'model': self.model.state_dict(), 'epoch': self.current_epoch, 'history': self.training_history}
# #         torch.save(state, path)


# import torch
# import torch.nn as nn
# import os
# from pathlib import Path

# class MelSpectrogramInpainter:
#     def __init__(self, hidden_channels, max_context_width, max_missing_width, model_type='fc_baseline', n_mels=128, device='cuda'):
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.model_type = model_type
#         self.n_mels = n_mels
        
#         # Norm Constants
#         self.min_db = -80.0
#         self.max_db = 0.0

#         if model_type == 'fc_baseline':
#             from models.fc_baseline import FCInpainter
#             # We use a context of 64 for the baseline to keep weights manageable
#             self.model = FCInpainter(n_mels=n_mels, context_width=max_context_width).to(self.device)
#         elif model_type == 'adaptive_unet':
#             from models.unet import AdaptiveUNet
#             self.model = AdaptiveUNet(n_mels=n_mels).to(self.device)
        
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

#         self.training_history = {'train_loss': [], 'val_loss': []}
#         self.current_epoch = 0
#         self.best_val_loss = float('inf')

#     def _norm(self, x): return (x - self.min_db) / (self.max_db - self.min_db)
#     def _denorm(self, x):
#         x = torch.clamp(x, 0.0, 1.0)
#         return x * (self.max_db - self.min_db) + self.min_db

#     def train_step(self, before, after, target):
#         self.model.train()
#         self.optimizer.zero_grad()
        
#         # Normalize
#         before, after, target = self._norm(before), self._norm(after), self._norm(target)
#         # Concatenate on Channel dimension (dim 1)
#         # Input to model: [B, 2, n_mels, context_width]
#         x_input = torch.cat([before, after], dim=1)

#         # Forward (passing target.shape[-1] handles the variable width)
#         target_width=target.shape[-1]
#         pred = self.model(x_input, target_width=target_width)

#         loss = self.criterion(pred, target)
#         loss.backward()
#         self.optimizer.step()
        
#         return loss.item(), {'mse': loss.item()}, pred

#     def validate_step(self, before, after, target):
#         self.model.eval()
#         with torch.no_grad():
#             before, after, target = self._norm(before), self._norm(after), self._norm(target)
#             x_input = torch.cat([before, after], dim=1)
#             pred = self.model(x_input, target.shape[-1])
#             loss = self.criterion(pred, target)
#         return loss.item(), pred

#     def predict(self, before, after, target_width):
#         self.model.eval()
#         with torch.no_grad():
#             x_input = torch.cat([self._norm(before), self._norm(after)], dim=1)
#             pred = self.model(x_input, target_width)
#             return self._denorm(pred)

#     def save_checkpoint(self, path, is_best=False):
#         state = {'model': self.model.state_dict(), 'epoch': self.current_epoch, 'history': self.training_history}
#         torch.save(state, path)

import torch
import torch.nn as nn

class MelSpectrogramInpainter:
    """
    Wrapper for a model that inpaints Mel-spectrograms.
    Inputs are already normalized in [-1, 1], so no internal normalization is needed.
    """
    def __init__(self, hidden_channels, max_context_width, max_missing_width, 
                 model_type='fc_baseline', n_mels=128, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.n_mels = n_mels

        if model_type == 'fc_baseline':
            from models.fc_baseline import FCInpainter
            self.model = FCInpainter(n_mels=n_mels, context_width=max_context_width).to(self.device)
        elif model_type == 'adaptive_unet':
            from models.unet import AdaptiveUNet
            self.model = AdaptiveUNet(n_mels=n_mels, hidden_channels=hidden_channels, context_width=max_context_width, missing_width=max_missing_width).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.training_history = {'train_loss': [], 'val_loss': []}
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    # -------------------------------
    # TRAIN / VALIDATION STEP
    # -------------------------------
    def train_step(self, before, after, target):
        """
        before, after, target: tensors in [-1, 1]
        before/after shape: [B, 1, n_mels, context_width]
        target shape: [B, 1, n_mels, target_width]
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Concatenate before & after along channel dimension
        x_input = torch.cat([before, after], dim=1)  # [B, 2, n_mels, context_width]

        target_width = target.shape[-1]
        pred = self.model(x_input)

        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'mse': loss.item()}, pred

    def validate_step(self, before, after, target):
        self.model.eval()
        with torch.no_grad():
            x_input = torch.cat([before, after], dim=1)
            pred = self.model(x_input)
            loss = self.criterion(pred, target)
        return loss.item(), pred

    # -------------------------------
    # PREDICTION
    # -------------------------------
    def predict(self, before, after):
        self.model.eval()
        with torch.no_grad():
            x_input = torch.cat([before, after], dim=1)
            pred = self.model(x_input)
            return pred  # already in [-1, 1]

    # -------------------------------
    # CHECKPOINTING
    # -------------------------------
    def save_checkpoint(self, path, is_best=False):
        state = {
            'model': self.model.state_dict(),
            'epoch': self.current_epoch,
            'history': self.training_history
        }
        torch.save(state, path)
