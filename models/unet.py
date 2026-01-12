import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AdaptiveUNet(nn.Module):
    """
    U-Net + learned temporal projection
    Context width -> Missing width
    """

    def __init__(
        self,
        n_mels=128,
        hidden_channels=64,
        context_width=1000,
        missing_width=128,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.context_width = context_width
        self.missing_width = missing_width

        # ---------------- Encoder ----------------
        self.enc1 = DoubleConv(2, hidden_channels)
        self.enc2 = DoubleConv(hidden_channels, hidden_channels * 2)
        self.enc3 = DoubleConv(hidden_channels * 2, hidden_channels * 4)

        # ---------------- Bottleneck ----------------
        self.bottleneck = DoubleConv(hidden_channels * 4, hidden_channels * 8)

        # ---------------- Decoder ----------------
        self.up3 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, 2, 2)
        self.dec3 = DoubleConv(hidden_channels * 8, hidden_channels * 4)

        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 2, 2)
        self.dec2 = DoubleConv(hidden_channels * 4, hidden_channels * 2)

        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 2, 2)
        self.dec1 = DoubleConv(hidden_channels * 2, hidden_channels)

        # ---------------- Output ----------------
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # projection temporelle apprise
        self.time_proj = nn.Conv1d(
            in_channels=context_width,
            out_channels=missing_width,
            kernel_size=1
        )

    def forward(self, x):
        """
        x : [B, 2, n_mels, context_width]
        return : [B, 1, n_mels, missing_width]
        """

        # ---------------- Encoder ----------------
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        # ---------------- Bottleneck ----------------
        b = self.bottleneck(p3)

        # ---------------- Decoder ----------------
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        # ---------------- Context Output ----------------
        out = self.final_conv(d1)      # [B, 1, n_mels, W_context]
        out = torch.tanh(out)

        # ---------------- Learned Temporal Projection ----------------
        x = out.squeeze(1)             # [B, n_mels, W_context]
        x = x.transpose(1, 2)          # [B, W_context, n_mels]
        x = self.time_proj(x)          # [B, W_missing, n_mels]
        x = x.transpose(1, 2)          # [B, n_mels, W_missing]
        x = x.unsqueeze(1)             # [B, 1, n_mels, W_missing]

        return x

def model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    size_fp32 = total_params * 4 / (1024 ** 2)   # en MB (float32)
    size_fp16 = total_params * 2 / (1024 ** 2)   # en MB (float16)
    size_fp4  = total_params * 0.5 / (1024 ** 2) # en MB (FP4 ≈ 4 bits)

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Model size FP32      : {size_fp32:.2f} MB")
    print(f"Model size FP16      : {size_fp16:.2f} MB")
    print(f"Model size FP4       : {size_fp4:.2f} MB")

model = AdaptiveUNet(
    n_mels=128,
    hidden_channels=64,
    context_width=1000,
    missing_width=128
)

model_size(model)
