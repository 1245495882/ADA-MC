import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

def autopad(k, p=None, d=1):
    """Calculate padding based on kernel size, padding specification, and dilation."""
    return (k - 1) // 2 if p is None else p

class ADAM(nn.Module):
    def __init__(self, channels, reduction_ratio=18, kernel_size=7):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(1, channels // reduction_ratio)

        # Shared Dimension Reduction
        self.dim_reduction = nn.Conv2d(channels, self.reduced_channels, kernel_size=1, bias=False)

        # Enhanced SE Block
        self.enhanced_se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Enhanced Spatial Attention Block using Deformable Convolution
        self.offset = nn.Conv2d(self.reduced_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=autopad(kernel_size))
        self.deform_conv = DeformConv2d(self.reduced_channels, self.reduced_channels, kernel_size=kernel_size, padding=autopad(kernel_size), bias=False)
        self.bn = nn.BatchNorm2d(self.reduced_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shared Dimension Restoration
        self.dim_restoration = nn.Conv2d(self.reduced_channels, channels, kernel_size=1, bias=False)

        # Adaptive Weight Fusion
        self.weight_se = nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight_sa = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        # Shared Dimension Reduction
        x_reduced = self.dim_reduction(x)

        # Enhanced SE Attention
        se_avg = self.enhanced_se_block(x_reduced)
        se_max = self.enhanced_se_block(torch.max(x_reduced, dim=1, keepdim=True)[0].expand_as(x_reduced))
        se_attn = se_avg + se_max

        # Enhanced Spatial Attention with Deformable Convolution
        offset = self.offset(x_reduced)  # 计算偏移量
        sa_attn = self.deform_conv(x_reduced, offset)  # 可变形卷积的输出
        sa_attn = self.bn(sa_attn)
        sa_attn = self.relu(sa_attn)
        sa_attn = torch.sigmoid(sa_attn)

        # Parallel Attention Application
        x_se = x_reduced * se_attn
        x_sa = x_reduced * sa_attn

        # Adaptive Fusion
        out = (self.weight_se * x_se + self.weight_sa * x_sa) / (self.weight_se + self.weight_sa + 1e-6)

        # Shared Dimension Restoration
        out = self.dim_restoration(out)
        return out

# Example of testing the ImprovedEnhancedAttention module
if __name__ == "__main__":
    input_channels = 64
    input_tensor = torch.randn(1, input_channels, 128, 128)  # Batch size 1, 64 channels, 128x128 size

    model = ADAM(input_channels)
    output = model(input_tensor)

    print("Output shape:", output.shape)  # Should output (1, input_channels, 128, 128)