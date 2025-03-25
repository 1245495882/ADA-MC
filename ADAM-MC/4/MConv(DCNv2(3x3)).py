import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class MConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c1 = c1 * 4

        # Independently handle the deformable convolutional layers for each part
        self.offset1 = nn.Conv2d(c1 // 4, 2 * k * k, kernel_size=k, stride=s, padding=autopad(k, p, d))
        self.conv1 = DeformConv2d(c1 // 4, c1 // 4, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.offset2 = nn.Conv2d(c1 // 4, 2 * k * k, kernel_size=k, stride=s, padding=autopad(k, p, d))
        self.conv2 = DeformConv2d(c1 // 4, c1 // 4, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.offset3 = nn.Conv2d(c1 // 4, 2 * k * k, kernel_size=k, stride=s, padding=autopad(k, p, d))
        self.conv3 = DeformConv2d(c1 // 4, c1 // 4, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.offset4 = nn.Conv2d(c1 // 4, 2 * k * k, kernel_size=k, stride=s, padding=autopad(k, p, d))
        self.conv4 = DeformConv2d(c1 // 4, c1 // 4, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)

        # The final convolutional layer after merging
        self.final_conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)

        self.bn = nn.BatchNorm2d(c1)  # BatchNorm will be used for the finally merged features
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Split the input into four parts
        x1 = x[..., ::2, ::2]  # One - quarter
        x2 = x[..., 1::2, ::2]  # One - quarter
        x3 = x[..., ::2, 1::2]  # One - quarter
        x4 = x[..., 1::2, 1::2]  # One - quarter

        # Calculate the offsets and perform deformable convolution
        offset1 = self.offset1(x1)
        x1 = self.conv1(x1, offset1)
        offset2 = self.offset2(x2)
        x2 = self.conv2(x2, offset2)
        offset3 = self.offset3(x3)
        x3 = self.conv3(x3, offset3)
        offset4 = self.offset4(x4)
        x4 = self.conv4(x4, offset4)

        # Concatenate the features
        x = torch.cat([x1, x2, x3, x4], 1)
        # Add a residual connection
        x_res = self.bn(x)
        x = x + x_res  # Residual connection

        # Activation
        x = self.act(x)
        # Channel adjustment
        x = self.final_conv(x)

        return x

if __name__ == "__main__":
    # test MConv
    c1 = 64  # input channels
    c2 = 128  # output channels
    input_tensor = torch.randn(1, 64, 128, 128)  # Batch size 1, c1channels, 64x64 size

    model = MConv(c1, c2)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # ouput is (1, 128, 64, 64)
