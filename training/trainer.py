# import torch
# from torch.utils.data import DataLoader
# from pathlib import Path
# import time

# class Trainer:
#     """Handles model training with checkpointing"""
    
#     def __init__(self, model, device='cuda'):
#         self.model = model
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
#     def create_dataloader(self, dataset, batch_size=8, shuffle=True):
        
#         def pad_collate(batch):
#             # batch is list of tuples: (before, after, target)
#             # each element is [n_mels, time]
            
#             before_list = [item[0] for item in batch]
#             after_list = [item[1] for item in batch]
#             target_list = [item[2] for item in batch]
            
#             # Since we fixed max_context_width in Dataset, before/after widths should be constant
#             # But we stack them anyway to be safe
#             before_batch = torch.stack(before_list) # [B, n_mels, W]
#             after_batch = torch.stack(after_list)   # [B, n_mels, W]
            
#             # Target might have variable width if multiple durations are used
#             max_target_w = max([t.shape[-1] for t in target_list])
            
#             padded_targets = []
#             for t in target_list:
#                 pad = max_target_w - t.shape[-1]
#                 if pad > 0:
#                     # Pad target on right
#                     t = torch.nn.functional.pad(t, (0, pad))
#                 padded_targets.append(t)
            
#             target_batch = torch.stack(padded_targets)
            
#             # Add channel dimension: [B, 1, n_mels, W]
#             before_batch = before_batch.unsqueeze(1)
#             after_batch = after_batch.unsqueeze(1)
#             target_batch = target_batch.unsqueeze(1)
            
#             return before_batch, after_batch, target_batch
        
#         return DataLoader(
#             dataset, batch_size=batch_size, shuffle=shuffle,
#             collate_fn=pad_collate, num_workers=0,
#             pin_memory=self.device.type == 'cuda'
#         )
    
#     def train_epoch(self, train_loader):
#         self.model.model.train()
#         total_loss = 0.0
        
#         for batch_idx, (before, after, target) in enumerate(train_loader):
#             before = before.to(self.device)
#             after = after.to(self.device)
#             target = target.to(self.device)
            
#             # Pass separated parts to model
#             loss, loss_dict, _ = self.model.train_step(before, after, target)
#             total_loss += loss
            
#             if batch_idx % 10 == 0:
#                 print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss:.6f}")
        
#         return total_loss / len(train_loader)
    
#     def validate(self, val_loader):
#         self.model.model.eval()
#         total_loss = 0.0
        
#         with torch.no_grad():
#             for before, after, target in val_loader:
#                 before = before.to(self.device)
#                 after = after.to(self.device)
#                 target = target.to(self.device)
                
#                 # In validation, we might want to predict exact target width
#                 # But target batch is padded to max width. 
#                 # Ideally, we pass the max width of this batch
#                 batch_target_width = target.shape[-1]
                
#                 pred = self.model.predict(before, after, batch_target_width)
                
#                 loss, _ = self.model.compute_loss(pred, target)
#                 total_loss += loss.item()
        
#         return total_loss / len(val_loader) if val_loader else 0.0

#     # ... (Rest of the train loop remains mostly the same) ...
#     def train(self, train_dataset, val_dataset=None, epochs=50, batch_size=8,
#               checkpoint_dir='checkpoints', checkpoint_freq=5):
#         """Main training loop"""
#         checkpoint_dir = Path(checkpoint_dir)
#         checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
#         train_loader = self.create_dataloader(train_dataset, batch_size, shuffle=True)
#         val_loader = None
#         if val_dataset:
#             val_loader = self.create_dataloader(val_dataset, batch_size, shuffle=False)
        
#         print(f"\nStarting training on {self.device}...")
        
#         for epoch in range(self.model.current_epoch, self.model.current_epoch + epochs):
#             start_time = time.time()
            
#             # Train
#             train_loss = self.train_epoch(train_loader)
#             self.model.training_history['train_loss'].append(train_loss)
            
#             # Validate
#             val_loss = 0.0
#             if val_loader:
#                 val_loss = self.validate(val_loader)
#                 self.model.training_history['val_loss'].append(val_loss)
#                 self.model.scheduler.step(val_loss)
                
#                 is_best = val_loss < self.model.best_val_loss
#                 if is_best:
#                     self.model.best_val_loss = val_loss
#             else:
#                 self.model.scheduler.step(train_loss)
            
#             print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} - {time.time()-start_time:.1f}s")
            
#             self.model.current_epoch = epoch + 1
            
#             if (epoch + 1) % checkpoint_freq == 0:
#                 self.model.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth", is_best=(val_loader and is_best))
        
#         self.model.save_checkpoint(checkpoint_dir / "model_final.pth")
#         return self.model.training_history


import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Trainer:
    """Handles model training with checkpointing. Inputs are already normalized in [-1,1]."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------
    # DATALOADER
    # -------------------------------
    def create_dataloader(self, dataset, batch_size=8, shuffle=True):

        def pad_collate(batch):
            before_list = [item[0] for item in batch]
            after_list  = [item[1] for item in batch]
            target_list = [item[2] for item in batch]

            before_batch = torch.stack(before_list)
            after_batch  = torch.stack(after_list)

            max_target_w = max(t.shape[-1] for t in target_list)
            padded_targets = []
            for t in target_list:
                pad = max_target_w - t.shape[-1]
                if pad > 0:
                    t = F.pad(t, (0, pad))
                padded_targets.append(t)

            target_batch = torch.stack(padded_targets)

            # [B, 1, n_mels, W]
            before_batch = before_batch.unsqueeze(1)
            after_batch  = after_batch.unsqueeze(1)
            target_batch = target_batch.unsqueeze(1)

            # ⚠️ CPU ONLY
            return before_batch, after_batch, target_batch

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pad_collate,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

    
    # -------------------------------
    # TRAINING / VALIDATION
    # -------------------------------
    def train_epoch(self, train_loader):
        self.model.model.train()
        total_loss = 0.0
        
        for batch_idx, (before, after, target) in enumerate(train_loader):
            before = before.to(self.device, non_blocking=True)
            after  = after.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Forward + backward handled inside model
            loss, loss_dict, _ = self.model.train_step(before, after, target)
            total_loss += loss
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss:.6f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for before, after, target in val_loader:
                before = before.to(self.device, non_blocking=True)
                after  = after.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                loss, pred = self.model.validate_step(before, after, target)
                total_loss += loss
        
        return total_loss / len(val_loader) if val_loader else 0.0

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    def train(self, train_dataset, val_dataset=None, epochs=50, batch_size=8,
              checkpoint_dir='checkpoints', checkpoint_freq=5):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        train_loader = self.create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = None
        if val_dataset:
            val_loader = self.create_dataloader(val_dataset, batch_size, shuffle=False)
        
        print(f"\nStarting training on {self.device}...")
        
        for epoch in range(self.model.current_epoch, self.model.current_epoch + epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            self.model.training_history['train_loss'].append(train_loss)
            
            val_loss = 0.0
            is_best = False
            if val_loader:
                val_loss = self.validate(val_loader)
                self.model.training_history['val_loss'].append(val_loss)
                self.model.scheduler.step(val_loss)
                
                is_best = val_loss < self.model.best_val_loss
                if is_best:
                    self.model.best_val_loss = val_loss
            else:
                self.model.scheduler.step(train_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} - {time.time()-start_time:.1f}s")
            
            self.model.current_epoch = epoch + 1
            
            if (epoch + 1) % checkpoint_freq == 0:
                self.model.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth", is_best=(val_loader and is_best))
        
        self.model.save_checkpoint(checkpoint_dir / "model_final.pth")
        
        # Plot history
        self.plot_history()
        return self.model.training_history

    # -------------------------------
    # HISTORY PLOT
    # -------------------------------
    def plot_history(self, save_path='training_history.png'):
        history = self.model.training_history
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        if history['val_loss']:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        plt.title('Model Training History (Normalized [-1,1] scale)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"History plot saved to {save_path}")
        plt.show()
