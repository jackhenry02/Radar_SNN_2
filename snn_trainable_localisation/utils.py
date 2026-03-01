"""Utility helpers for SNN localisation training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset

from simulate_physics import simulate_binaural_echo
from spike_encoder import FilterbankSpikeEncoder


@dataclass
class TargetNorm:
    """Normalization constants for targets."""

    theta_deg: float = 90.0
    range_m: float = 30.0


def set_seed(seed: int = 0) -> None:
    """Set torch RNG seed for reproducibility."""
    torch.manual_seed(seed)


def normalize_targets(theta_deg: float, r_m: float, norm: TargetNorm) -> torch.Tensor:
    """Normalize targets to roughly [-1, 1] range."""
    return torch.tensor([theta_deg / norm.theta_deg, r_m / norm.range_m], dtype=torch.float32)


def denormalize_targets(y: torch.Tensor, norm: TargetNorm) -> torch.Tensor:
    """Denormalize from network outputs back to physical units."""
    theta = y[..., 0] * norm.theta_deg
    r = y[..., 1] * norm.range_m
    return torch.stack([theta, r], dim=-1)


def compute_errors(
    pred_norm: torch.Tensor, target_norm: torch.Tensor, norm: TargetNorm
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute absolute errors in degrees and meters."""
    pred = denormalize_targets(pred_norm, norm)
    target = denormalize_targets(target_norm, norm)
    theta_err = (pred[..., 0] - target[..., 0]).abs()
    range_err = (pred[..., 1] - target[..., 1]).abs()
    return theta_err, range_err


class EchoDataset(Dataset):
    """Dataset that generates or caches spike-encoded echoes."""

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len_ms: int = 200,
        n_channels: int = 32,
        seed: int = 0,
        cache: bool = True,
        noise_std: float = 0.01,
        theta_range: Tuple[float, float] = (-90.0, 90.0),
        r_range: Tuple[float, float] = (5.0, 30.0),
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.seq_len_ms = seq_len_ms
        self.n_channels = n_channels
        self.seed = seed
        self.cache = cache
        self.noise_std = noise_std
        self.theta_range = theta_range
        self.r_range = r_range
        self.norm = TargetNorm()

        self.encoder = FilterbankSpikeEncoder(n_channels=n_channels)
        self.encoder.eval()

        # Pre-sample targets for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        theta_span = theta_range[1] - theta_range[0]
        r_span = r_range[1] - r_range[0]
        self.theta_vals = theta_range[0] + theta_span * torch.rand(n_samples, generator=g)
        self.r_vals = r_range[0] + r_span * torch.rand(n_samples, generator=g)

        self._cache_spikes = None
        self._cache_targets = None
        if self.cache:
            self._precompute_cache()

    def _precompute_cache(self) -> None:
        spikes = torch.empty(
            self.n_samples,
            self.seq_len_ms,
            self.n_channels * 2,
            dtype=torch.uint8,
        )
        targets = torch.empty(self.n_samples, 2, dtype=torch.float32)

        with torch.no_grad():
            for i in range(self.n_samples):
                theta = float(self.theta_vals[i].item())
                r_m = float(self.r_vals[i].item())

                left, right = simulate_binaural_echo(
                    theta_deg=theta,
                    r_m=r_m,
                    seq_len_ms=self.seq_len_ms,
                    noise_std=self.noise_std,
                )

                # Encode to spikes
                spk, _, _ = self.encoder(left.unsqueeze(0), right.unsqueeze(0))
                spk = spk.squeeze(1)  # [T, channels*2]
                spikes[i] = spk.to(dtype=torch.uint8)
                targets[i] = normalize_targets(theta, r_m, self.norm)

        self._cache_spikes = spikes
        self._cache_targets = targets

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache and self._cache_spikes is not None:
            spk = self._cache_spikes[idx].float()
            target = self._cache_targets[idx]
            return spk, target

        # Fallback: on-the-fly generation
        theta = float(self.theta_vals[idx].item())
        r_m = float(self.r_vals[idx].item())
        left, right = simulate_binaural_echo(
            theta_deg=theta,
            r_m=r_m,
            seq_len_ms=self.seq_len_ms,
            noise_std=self.noise_std,
        )
        spk, _, _ = self.encoder(left.unsqueeze(0), right.unsqueeze(0))
        spk = spk.squeeze(1).float()
        target = normalize_targets(theta, r_m, self.norm)
        return spk, target
