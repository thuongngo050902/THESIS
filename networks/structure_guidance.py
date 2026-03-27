import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import persistence
from networks.basic_module import Conv2dLayer


@persistence.persistent_class
class StructureInputBuilder(nn.Module):
    def forward(self, image, mask):
        visible = image * mask
        gray = visible.mean(dim=1, keepdim=True)
        grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        boundary = torch.abs(mask - F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1))
        return torch.cat([visible, gray, grad_x, grad_y, boundary], dim=1), boundary


@persistence.persistent_class
class StructureEncoder(nn.Module):
    def __init__(self, in_channels=7, base_channels=64, out_dim=180):
        super().__init__()
        self.stem = Conv2dLayer(in_channels=in_channels, out_channels=base_channels, kernel_size=3, activation='lrelu')
        self.refine = Conv2dLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, activation='lrelu')
        self.down_32 = Conv2dLayer(in_channels=base_channels, out_channels=base_channels * 2, kernel_size=3, down=2, activation='lrelu')
        self.down_16 = Conv2dLayer(in_channels=base_channels * 2, out_channels=base_channels * 4, kernel_size=3, down=2, activation='lrelu')
        self.to_32 = Conv2dLayer(in_channels=base_channels * 2, out_channels=out_dim, kernel_size=3, activation='linear')
        self.to_16 = Conv2dLayer(in_channels=base_channels * 4, out_channels=out_dim, kernel_size=3, activation='linear')

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.stem(x)
        x = self.refine(x)
        feat_32 = self.down_32(x)
        feat_16 = self.down_16(feat_32)
        return {
            32: self.to_32(feat_32),
            16: self.to_16(feat_16),
        }


@persistence.persistent_class
class StructureAwareAttentionBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.tran_struct_bias_proj = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.tran_struct_bias_proj.weight)
        nn.init.zeros_(self.tran_struct_bias_proj.bias)

    def forward(self, x):
        return self.tran_struct_bias_proj(self.norm(x))


@persistence.persistent_class
class StructureResidualAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.tran_struct_norm = nn.LayerNorm(dim)
        self.tran_struct_down = nn.Linear(dim, hidden_dim)
        self.tran_struct_up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.tran_struct_up.weight)
        nn.init.zeros_(self.tran_struct_up.bias)

    def forward(self, x):
        x = self.tran_struct_norm(x)
        x = self.tran_struct_down(x)
        x = F.gelu(x)
        x = self.tran_struct_up(x)
        return x


@persistence.persistent_class
class MaskSeverityGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, pooled_struct, pooled_boundary, pooled_mask):
        score = torch.cat([pooled_struct, pooled_boundary, pooled_mask], dim=1)
        return torch.sigmoid(self.fc(score))