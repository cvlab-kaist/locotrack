import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models import utils


class CMDTop(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_shapes, strides):
        super(CMDTop, self).__init__()
        self.in_channels = [in_channel] + list(out_channels[:-1])
        self.out_channels = out_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides

        self.conv = nn.ModuleList([
            nn.Sequential(
                utils.Conv2dSamePadding(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    kernel_size=self.kernel_shapes[i],
                    stride=self.strides[i],
                ),
                nn.GroupNorm(out_channels[i] // 16, out_channels[i]),
                nn.ReLU()
            ) for i in range(len(out_channels))
        ])

    def forward(self, x):
        """
        x: (b, h, w, i, j)
        """
        out1 = rearrange(x, 'b h w i j -> b (i j) h w')
        out2 = rearrange(x, 'b h w i j -> b (h w) i j')
        
        for i in range(len(self.out_channels)):
            out1 = self.conv[i](out1)
        
        for i in range(len(self.out_channels)):
            out2 = self.conv[i](out2)

        out1 = torch.mean(out1, dim=(2, 3)) # (b, out_channels[-1])
        out2 = torch.mean(out2, dim=(2, 3)) # (b, out_channels[-1])

        return torch.cat([out1, out2], dim=-1) # (b, 2*out_channels[-1])

