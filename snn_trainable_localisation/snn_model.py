"""Trainable SNN model for azimuth and range regression."""

from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate


def _beta_from_tau(tau_ms: float, dt_ms: float = 1.0) -> float:
    """Convert membrane time constant to decay factor beta."""
    return math.exp(-dt_ms / tau_ms)


class SNNLocalizer(nn.Module):
    """Two-layer LIF SNN with a linear readout.

    Input shape: [T, batch, channels]
    Output shape: [batch, 2] => [theta_norm, range_norm]
    """

    def __init__(
        self,
        in_features: int,
        hidden_mso: int = 128,
        hidden_ic: int = 64,
        tau_mso_ms: float = 5.0,
        tau_ic_ms: float = 20.0,
        thr_mso: float = 1.2,
        thr_ic: float = 1.0,
    ) -> None:
        super().__init__()

        beta_mso = _beta_from_tau(tau_mso_ms)
        beta_ic = _beta_from_tau(tau_ic_ms)

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(in_features, hidden_mso)
        self.lif1 = snn.Leaky(
            beta=beta_mso,
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )

        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=beta_ic,
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )

        self.readout = nn.Linear(hidden_ic, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass through time.

        Parameters
        ----------
        spikes : torch.Tensor
            Shape [T, batch, in_features]

        Returns
        -------
        out : torch.Tensor
            Shape [batch, 2]
        stats : dict
            Simple activity stats for logging.
        """
        assert spikes.ndim == 3, "Expected [T, batch, features]."
        t_steps, batch, _ = spikes.shape

        # Initialize membrane states
        mem1 = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        spk1_sum = 0.0
        spk2_sum = 0.0

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            spk1_sum += spk1.mean().item()
            spk2_sum += spk2.mean().item()

        # Temporal integration via membrane average
        mem2_avg = mem2_sum / float(t_steps)
        out = self.readout(mem2_avg)

        stats = {
            "spk1_mean": spk1_sum / float(t_steps),
            "spk2_mean": spk2_sum / float(t_steps),
        }
        return out, stats
