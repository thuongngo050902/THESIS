# Focal Frequency Loss for image reconstruction and synthesis.
# Based on: Jiang et al., "Focal Frequency Loss for Image Reconstruction
# and Synthesis", ICCV 2021. (https://arxiv.org/abs/2012.12821)

import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss.

    Computes the distance between predicted and target images in the
    frequency domain, with dynamic weighting that focuses training on
    hard-to-reconstruct frequency components.

    Args:
        alpha (float): Exponent for the focal weight. Higher values
            increase focus on hard frequencies. Default: 1.0 (original paper).
        log_matrix (bool): If True, use log(1 + distance) to stabilize
            the focal weight for very large spectral differences.
            Default: False (matches the original paper).
        patch_factor (int): If > 1, splits the image into non-overlapping
            patches and computes FFL per patch, then averages. This can
            capture local frequency information. Default: 1 (no patching,
            matches the original paper default).
    """

    def __init__(self, alpha=1.0, log_matrix=False, patch_factor=1):
        super().__init__()
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.patch_factor = patch_factor

    def _frequency_distance(self, pred_freq, target_freq):
        """Compute per-element distance between two complex frequency tensors.

        Args:
            pred_freq: Complex tensor [B, C, H, W].
            target_freq: Complex tensor [B, C, H, W].

        Returns:
            Distance tensor [B, C, H, W] (real-valued).
        """
        # Distance = sqrt((real_diff)^2 + (imag_diff)^2)
        diff = pred_freq - target_freq
        distance = torch.abs(diff)  # magnitude of complex difference
        return distance

    def forward(self, pred, target):
        """Compute Focal Frequency Loss.

        Args:
            pred (torch.Tensor): Predicted image [B, C, H, W] in [-1, 1].
            target (torch.Tensor): Target image [B, C, H, W] in [-1, 1].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize from [-1, 1] to [0, 1] for numerical stability in FFT.
        pred = (pred + 1.0) * 0.5
        target = (target + 1.0) * 0.5

        if self.patch_factor > 1:
            # Split into non-overlapping patches.
            _, _, H, W = pred.shape
            assert H % self.patch_factor == 0 and W % self.patch_factor == 0, \
                f"Image size ({H}, {W}) must be divisible by patch_factor ({self.patch_factor})"
            pH = H // self.patch_factor
            pW = W // self.patch_factor
            pred = pred.unfold(2, pH, pH).unfold(3, pW, pW)  # [B,C,nH,nW,pH,pW]
            target = target.unfold(2, pH, pH).unfold(3, pW, pW)
            pred = pred.contiguous().view(-1, pred.shape[1], pH, pW)  # [B*nH*nW, C, pH, pW]
            target = target.contiguous().view(-1, target.shape[1], pH, pW)

        # 2D FFT (full complex, not rfft — matches original paper).
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target.detach(), norm='ortho')

        # Frequency distance matrix.
        freq_distance = self._frequency_distance(pred_freq, target_freq)

        # Focal weight: w = distance^alpha (dynamic, focuses on hard frequencies).
        if self.log_matrix:
            weight = torch.log1p(freq_distance)  # log(1 + d) for stability
        else:
            weight = freq_distance  # d^1 when alpha=1

        if self.alpha != 1.0:
            weight = weight ** self.alpha

        # Weighted frequency loss.
        loss = weight * freq_distance  # w * d = d^(alpha+1) when not using log

        # Average over all dimensions.
        return loss.mean()
