"""Experimental architecture variants for rapid SNN viability search.

All models keep:
- MSO-like coincidence layer (LIF)
- IC-like integration layer (LIF)
- Optional higher integration layer (LIF)
- Non-spiking readout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def _beta_from_tau(tau_ms: float, dt_ms: float = 1.0) -> float:
    return math.exp(-dt_ms / tau_ms)


@dataclass
class ArchSpec:
    name: str
    description: str
    num_layers: int
    hidden_dims: Tuple[int, ...]
    tau_mem_values: Tuple[float, ...]


class BaseSNN(nn.Module):
    """Base class with helper for stats."""

    def _init_stats(self) -> Dict[str, float]:
        return {
            "spk1_mean": 0.0,
            "spk2_mean": 0.0,
            "spk3_mean": 0.0,
        }

    @staticmethod
    def _accumulate_stats(stats: Dict[str, float], key: str, value: float) -> None:
        stats[key] += value


class SNNBaseline(BaseSNN):
    """Baseline: 2-layer MSO->IC->readout."""

    def __init__(
        self,
        in_features: int,
        hidden_mso: int,
        hidden_ic: int,
        tau_mso_ms: float,
        tau_ic_ms: float,
        thr_mso: float,
        thr_ic: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.fc1 = nn.Linear(in_features, hidden_mso)
        self.lif1 = snn.Leaky(
            beta=_beta_from_tau(tau_mso_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_ic, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem1 = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        stats = self._init_stats()

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())

        mem2_avg = mem2_sum / float(t_steps)
        out = self.readout(mem2_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        return out, stats


class SNNWithSkipIC(BaseSNN):
    """Baseline plus skip from input to IC current."""

    def __init__(
        self,
        in_features: int,
        hidden_mso: int,
        hidden_ic: int,
        tau_mso_ms: float,
        tau_ic_ms: float,
        thr_mso: float,
        thr_ic: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.fc1 = nn.Linear(in_features, hidden_mso)
        self.lif1 = snn.Leaky(
            beta=_beta_from_tau(tau_mso_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.fc_skip = nn.Linear(in_features, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_ic, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem1 = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        stats = self._init_stats()

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1) + self.fc_skip(spikes[t])
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())

        mem2_avg = mem2_sum / float(t_steps)
        out = self.readout(mem2_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        return out, stats


class SNNDeep(BaseSNN):
    """Three spiking layers: MSO -> IC -> higher integration."""

    def __init__(
        self,
        in_features: int,
        hidden_mso: int,
        hidden_ic: int,
        hidden_hi: int,
        tau_mso_ms: float,
        tau_ic_ms: float,
        tau_hi_ms: float,
        thr_mso: float,
        thr_ic: float,
        thr_hi: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.fc1 = nn.Linear(in_features, hidden_mso)
        self.lif1 = snn.Leaky(
            beta=_beta_from_tau(tau_mso_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc3 = nn.Linear(hidden_ic, hidden_hi)
        self.lif3 = snn.Leaky(
            beta=_beta_from_tau(tau_hi_ms),
            threshold=thr_hi,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_hi, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem1 = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)
        mem3 = torch.zeros(batch, self.fc3.out_features, device=spikes.device)

        mem3_sum = torch.zeros_like(mem3)
        stats = self._init_stats()

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            mem3_sum += mem3
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())
            self._accumulate_stats(stats, "spk3_mean", spk3.mean().item())

        mem3_avg = mem3_sum / float(t_steps)
        out = self.readout(mem3_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        stats["spk3_mean"] /= float(t_steps)
        return out, stats


class SNNCoincidenceGated(BaseSNN):
    """MSO coincidence with left/right projections and multiplicative gating."""

    def __init__(
        self,
        in_features: int,
        hidden_mso: int,
        hidden_ic: int,
        tau_mso_ms: float,
        tau_ic_ms: float,
        thr_mso: float,
        thr_ic: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        assert in_features % 2 == 0, "Expected concatenated L/R channels."
        self.half = in_features // 2
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        self.fc_left = nn.Linear(self.half, hidden_mso)
        self.fc_right = nn.Linear(self.half, hidden_mso)
        self.gate = nn.Parameter(torch.ones(hidden_mso))

        self.lif1 = snn.Leaky(
            beta=_beta_from_tau(tau_mso_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_ic, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem1 = torch.zeros(batch, self.fc_left.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        stats = self._init_stats()

        for t in range(t_steps):
            left = spikes[t, :, : self.half]
            right = spikes[t, :, self.half :]

            cur_l = self.fc_left(left)
            cur_r = self.fc_right(right)
            # Coincidence gating term
            cur1 = cur_l + cur_r + (cur_l * cur_r) * self.gate

            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())

        mem2_avg = mem2_sum / float(t_steps)
        out = self.readout(mem2_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        return out, stats


class SNNParallelMSO(BaseSNN):
    """Parallel MSO subpopulations with different time constants."""

    def __init__(
        self,
        in_features: int,
        hidden_mso_total: int,
        hidden_ic: int,
        tau_mso_fast_ms: float,
        tau_mso_slow_ms: float,
        tau_ic_ms: float,
        thr_mso: float,
        thr_ic: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        h_fast = hidden_mso_total // 2
        h_slow = hidden_mso_total - h_fast

        self.fc_fast = nn.Linear(in_features, h_fast)
        self.fc_slow = nn.Linear(in_features, h_slow)

        self.lif_fast = snn.Leaky(
            beta=_beta_from_tau(tau_mso_fast_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.lif_slow = snn.Leaky(
            beta=_beta_from_tau(tau_mso_slow_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )

        self.fc2 = nn.Linear(hidden_mso_total, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_ic, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem_fast = torch.zeros(batch, self.fc_fast.out_features, device=spikes.device)
        mem_slow = torch.zeros(batch, self.fc_slow.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        stats = self._init_stats()

        for t in range(t_steps):
            cur_f = self.fc_fast(spikes[t])
            spk_f, mem_fast = self.lif_fast(cur_f, mem_fast)

            cur_s = self.fc_slow(spikes[t])
            spk_s, mem_slow = self.lif_slow(cur_s, mem_slow)

            spk1 = torch.cat([spk_f, spk_s], dim=-1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())

        mem2_avg = mem2_sum / float(t_steps)
        out = self.readout(mem2_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        return out, stats


class SNNResidualReadout(BaseSNN):
    """Residual from MSO to readout to improve gradient flow."""

    def __init__(
        self,
        in_features: int,
        hidden_mso: int,
        hidden_ic: int,
        tau_mso_ms: float,
        tau_ic_ms: float,
        thr_mso: float,
        thr_ic: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.fc1 = nn.Linear(in_features, hidden_mso)
        self.lif1 = snn.Leaky(
            beta=_beta_from_tau(tau_mso_ms),
            threshold=thr_mso,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_mso, hidden_ic)
        self.lif2 = snn.Leaky(
            beta=_beta_from_tau(tau_ic_ms),
            threshold=thr_ic,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
        )
        self.readout = nn.Linear(hidden_ic, 2)
        self.skip_readout = nn.Linear(hidden_mso, 2)

    def forward(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        t_steps, batch, _ = spikes.shape
        mem1 = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=spikes.device)

        mem2_sum = torch.zeros_like(mem2)
        spk1_sum = torch.zeros(batch, self.fc1.out_features, device=spikes.device)
        stats = self._init_stats()

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_sum += mem2
            spk1_sum += spk1
            self._accumulate_stats(stats, "spk1_mean", spk1.mean().item())
            self._accumulate_stats(stats, "spk2_mean", spk2.mean().item())

        mem2_avg = mem2_sum / float(t_steps)
        spk1_avg = spk1_sum / float(t_steps)
        out = self.readout(mem2_avg) + self.skip_readout(spk1_avg)

        stats["spk1_mean"] /= float(t_steps)
        stats["spk2_mean"] /= float(t_steps)
        return out, stats
