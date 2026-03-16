# Focal Frequency Loss for image reconstruction and synthesis.
# Based on: Jiang et al., "Focal Frequency Loss for Image Reconstruction
# and Synthesis", ICCV 2021. (https://arxiv.org/abs/2012.12821)

import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """A stabilized Focal Frequency Loss close to the official formulation."""

    def __init__(self, alpha=1.0, log_matrix=False, patch_factor=1, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.patch_factor = patch_factor
        self.eps = eps

    def _to_patches(self, x):
        if self.patch_factor <= 1:
            return x
        _, _, height, width = x.shape
        if height % self.patch_factor != 0 or width % self.patch_factor != 0:
            raise ValueError(
                f"Image size ({height}, {width}) must be divisible by patch_factor ({self.patch_factor})"
            )
        patch_h = height // self.patch_factor
        patch_w = width // self.patch_factor
        x = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        return x.contiguous().view(-1, x.shape[1], patch_h, patch_w)

    def _compute_weight(self, pred_freq, target_freq):
        diff = pred_freq - target_freq
        distance = diff.real.square() + diff.imag.square()
        weight = torch.sqrt(distance + self.eps).pow(self.alpha)
        if self.log_matrix:
            weight = torch.log1p(weight)
        weight = weight / weight.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        weight = torch.clamp(weight, min=0.0, max=1.0).detach()
        return weight, distance

    def forward(self, pred, target):
        pred = ((pred + 1.0) * 0.5).to(torch.float32)
        target = ((target + 1.0) * 0.5).detach().to(torch.float32)

        pred = self._to_patches(pred)
        target = self._to_patches(target)

        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')
        weight, distance = self._compute_weight(pred_freq, target_freq)
        loss = weight * distance
        return loss.mean()
