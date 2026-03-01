"""Spike encoder with a simple gammatone-like filterbank.

The encoder runs at 1 ms resolution and outputs spikes with shape:
    [T, batch, channels * 2]
"""

from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterbankSpikeEncoder(nn.Module):
    """Bandpass filterbank + rectification + thresholding.

    This is a simplified, stable approximation designed for low
    time resolution (1 ms) and small channel counts.
    """

    def __init__(
        self,
        n_channels: int = 32,
        kernel_ms: int = 25,
        dt_ms: int = 1,
        f_min_hz: float = 20.0,
        f_max_hz: float = 400.0,
        threshold: float = 1.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        assert dt_ms == 1, "This project assumes 1 ms time resolution."
        self.n_channels = n_channels
        self.kernel_ms = kernel_ms
        self.dt_ms = dt_ms
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        self.threshold = threshold
        self.normalize = normalize

        # Precompute and register filterbank kernels
        kernels = self._create_filterbank()
        # Shape: [channels, 1, kernel]
        self.register_buffer("kernels", kernels)

    def _create_filterbank(self) -> torch.Tensor:
        """Create gammatone-like FIR kernels for each channel."""
        t = torch.arange(self.kernel_ms, dtype=torch.float32) * 1e-3
        freqs = torch.linspace(self.f_min_hz, self.f_max_hz, self.n_channels)

        kernels = []
        for f in freqs:
            # Gammatone-like: t^3 * exp(-2*pi*b*t) * cos(2*pi*f*t)
            # Use a lightweight bandwidth rule for stability
            bandwidth = 10.0 + 0.1 * float(f)
            env = (t ** 3) * torch.exp(-2.0 * math.pi * bandwidth * t)
            carrier = torch.cos(2.0 * math.pi * float(f) * t)
            k = env * carrier
            # Normalize energy to keep thresholds consistent
            k = k / (k.abs().sum() + 1e-6)
            kernels.append(k)

        kernel_bank = torch.stack(kernels, dim=0).unsqueeze(1)
        return kernel_bank

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode left/right waveforms into spike trains.

        Parameters
        ----------
        left, right : torch.Tensor
            Shape [batch, T]

        Returns
        -------
        spikes : torch.Tensor
            Shape [T, batch, channels * 2]
        left_filt, right_filt : torch.Tensor
            Filtered analog features, shape [batch, channels, T]
        """
        assert left.ndim == 2 and right.ndim == 2, "Expected [batch, T]."
        batch, t_len = left.shape

        # Convolution expects [batch, channels_in, T]
        left_in = left.unsqueeze(1)
        right_in = right.unsqueeze(1)

        # Causal-ish filtering: pad at start so output length matches input
        pad = self.kernel_ms - 1
        left_f = F.conv1d(left_in, self.kernels, padding=pad)
        right_f = F.conv1d(right_in, self.kernels, padding=pad)

        # Trim to original length
        left_f = left_f[:, :, :t_len]
        right_f = right_f[:, :, :t_len]

        # Rectify
        left_f = torch.relu(left_f)
        right_f = torch.relu(right_f)

        # Normalize per channel to stabilize thresholds
        if self.normalize:
            left_f = left_f / (left_f.mean(dim=-1, keepdim=True) + 1e-6)
            right_f = right_f / (right_f.mean(dim=-1, keepdim=True) + 1e-6)

        # Threshold to spikes
        spk_left = (left_f > self.threshold).float()
        spk_right = (right_f > self.threshold).float()

        # Concatenate ears along channel dimension
        spikes = torch.cat([spk_left, spk_right], dim=1)
        # Reorder to [T, batch, channels*2]
        spikes = spikes.permute(2, 0, 1).contiguous()

        return spikes, left_f, right_f


if __name__ == "__main__":
    # Quick sanity check with dummy waves
    dummy = torch.zeros(2, 200)
    dummy[:, 50:60] = 1.0
    enc = FilterbankSpikeEncoder(n_channels=8)
    spk, l_f, r_f = enc(dummy, dummy)
    print("Spikes:", spk.shape, "Analog:", l_f.shape)
