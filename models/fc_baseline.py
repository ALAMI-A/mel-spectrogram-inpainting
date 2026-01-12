import torch
import torch.nn as nn
import torch.nn.functional as F

class FCInpainter(nn.Module):
    """
    Baseline: Fully Connected Network.
    Treats the spectrogram as a flat list of pixels.
    """
    def __init__(self, n_mels=128, context_width=1024, internal_gap_width=128, internal_size=1024):
        super().__init__()
        self.n_mels = n_mels
        self.context_width = context_width
        self.internal_gap_width = internal_gap_width
        
        # Input: 2 Context windows (Before + After)
        input_dim = 2 * n_mels * context_width
        # Output: 1 Gap window
        output_dim = n_mels * internal_gap_width
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, internal_size),
            nn.ReLU(),
            nn.BatchNorm1d(internal_size),
            nn.Linear(internal_size, internal_size),
            nn.ReLU(),
            nn.Linear(internal_size, output_dim),
            nn.Sigmoid() # Force output to [0, 1] for stability
        )

    def forward(self, before, after, target_width):
        """
        Args:
            before: [B, 1, n_mels, context_width]
            after:  [B, 1, n_mels, context_width]
            target_width: The actual number of frames needed (e.g., 129)
        """
        # 1. Combine contexts
        x = torch.cat([before, after], dim=1) # [B, 2, n_mels, context_width]
        
        # 2. Forward pass
        out = self.network(x) # [B, n_mels * internal_gap_width]
        
        # 3. Reshape to image-like format
        out = out.view(-1, 1, self.n_mels, self.internal_gap_width)
        
        # 4. ADAPTIVE RESIZING (Fixes the 128 vs 129 error)
        # We interpolate the time dimension to match exactly what the audio needs
        if out.shape[-1] != target_width:
            out = F.interpolate(out, size=(self.n_mels, target_width), mode='bilinear', align_corners=False)
            
        return out
    
def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    size_fp32 = total_params * 4 / (1024 ** 2)   # MB
    size_fp16 = total_params * 2 / (1024 ** 2)   # MB

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Model size (FP32)    : {size_fp32:.2f} MB")
    print(f"Model size (FP16)    : {size_fp16:.2f} MB")

model = FCInpainter(
    n_mels=128,
    context_width=1024,
    internal_gap_width=128,
    internal_size=1024
)

print_model_size(model)