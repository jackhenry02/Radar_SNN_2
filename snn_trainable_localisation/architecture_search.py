"""Exploratory architecture search for SNN localization.

Phase 1 (Attempts 1-6):
- 5 epochs per trial, max 5 Optuna trials per architecture.
- Mark architecture VIABLE if MAE_theta < 15 deg AND MAE_range < 6 m.
- Mark NON-VIABLE if after 5 epochs MAE_theta > 30 deg OR MAE_range > 10 m.

Phase 2 (Attempts 7-10):
- Only VIABLE architectures from Phase 1.
- 20 Optuna trials per architecture, 15 epochs per trial.
- No architectural changes (only hyperparameters).

Logs:
- architecture_search_log.csv (per attempt)
- architecture_trials_log.csv (per trial)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Optuna is not installed. Install it with: pip install optuna") from exc

from arch_models import (
    ArchSpec,
    SNNBaseline,
    SNNWithSkipIC,
    SNNDeep,
    SNNCoincidenceGated,
    SNNParallelMSO,
    SNNResidualReadout,
)
from utils import EchoDataset, TargetNorm, compute_errors, set_seed


ARCH_LOG = os.path.join(THIS_DIR, "architecture_search_log.csv")
TRIAL_LOG = os.path.join(THIS_DIR, "architecture_trials_log.csv")


@dataclass
class SearchConfig:
    n_samples: int = 1000
    seq_len_ms: int = 200
    n_channels: int = 32
    batch_size: int = 16
    val_split: float = 0.2
    noise_std: float = 0.005

    phase1_epochs: int = 5
    phase2_epochs: int = 15

    phase1_trials: int = 5
    phase2_trials: int = 20

    grad_clip: float = 1.0
    seed: int = 42
    use_mps_if_available: bool = True


def _select_device(use_mps: bool) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_time_major(spikes: torch.Tensor) -> torch.Tensor:
    return spikes.permute(1, 0, 2).contiguous()


def _grad_status(grad_norms: List[float]) -> str:
    if not grad_norms:
        return "unknown"
    if any(math.isnan(g) or math.isinf(g) for g in grad_norms):
        return "exploding"
    max_g = max(grad_norms)
    mean_g = sum(grad_norms) / float(len(grad_norms))
    if max_g > 100.0:
        return "exploding"
    if mean_g < 1e-6:
        return "vanishing"
    return "ok"


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    norm: TargetNorm,
    grad_clip: float,
) -> Tuple[float, float, float, str]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_theta = 0.0
    total_range = 0.0
    count = 0

    grad_norms: List[float] = []
    criterion = nn.MSELoss()

    for spikes, targets in loader:
        spikes = spikes.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        spikes = _batch_to_time_major(spikes)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs, _ = model(spikes)
        loss = criterion(outputs, targets)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Record gradient norm for diagnostics
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
            grad_norms.append(math.sqrt(total_norm))

            optimizer.step()

        theta_err, range_err = compute_errors(outputs.detach(), targets, norm)

        total_loss += float(loss.item()) * spikes.size(1)
        total_theta += float(theta_err.mean().item()) * spikes.size(1)
        total_range += float(range_err.mean().item()) * spikes.size(1)
        count += spikes.size(1)

    if count == 0:
        return math.inf, math.inf, math.inf, "unknown"

    return (
        total_loss / count,
        total_theta / count,
        total_range / count,
        _grad_status(grad_norms),
    )


def _detect_divergence(loss: float) -> bool:
    if math.isnan(loss) or math.isinf(loss):
        return True
    if loss > 1e3:
        return True
    return False


def _eval_spike_rates(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    spk1_vals: List[float] = []
    spk2_vals: List[float] = []
    with torch.no_grad():
        for spikes, _ in loader:
            spikes = spikes.to(device=device, dtype=torch.float32)
            spikes = _batch_to_time_major(spikes)
            _, stats = model(spikes)
            spk1_vals.append(float(stats.get("spk1_mean", 0.0)))
            spk2_vals.append(float(stats.get("spk2_mean", 0.0)))
    if not spk1_vals:
        return 0.0, 0.0
    return sum(spk1_vals) / len(spk1_vals), sum(spk2_vals) / len(spk2_vals)


def _init_logs() -> None:
    if not os.path.exists(ARCH_LOG):
        with open(ARCH_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Attempt_Number",
                    "Phase",
                    "Architecture_Description",
                    "Num_Layers",
                    "Hidden_Dims",
                    "Tau_mem_values",
                    "Spike_Rate_L1",
                    "Spike_Rate_L2",
                    "Validation_MAE_theta",
                    "Validation_MAE_range",
                    "Viable (Y/N)",
                    "Comments",
                ]
            )

    if not os.path.exists(TRIAL_LOG):
        with open(TRIAL_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Attempt_Number",
                    "Phase",
                    "Trial",
                    "Val_Loss",
                    "Val_MAE_theta",
                    "Val_MAE_range",
                    "Params_JSON",
                    "Grad_Status",
                    "Comments",
                ]
            )


def _append_arch_log(row: List[Any]) -> None:
    with open(ARCH_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _append_trial_log(row: List[Any]) -> None:
    with open(TRIAL_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _build_arch_list() -> List[Tuple[ArchSpec, Any]]:
    return [
        (
            ArchSpec(
                name="baseline",
                description="Baseline MSO->IC with LIF layers",
                num_layers=2,
                hidden_dims=(128, 64),
                tau_mem_values=(5.0, 20.0),
            ),
            SNNBaseline,
        ),
        (
            ArchSpec(
                name="skip_ic",
                description="MSO->IC with skip from input to IC current",
                num_layers=2,
                hidden_dims=(128, 64),
                tau_mem_values=(5.0, 20.0),
            ),
            SNNWithSkipIC,
        ),
        (
            ArchSpec(
                name="deep",
                description="MSO->IC->higher integration layer",
                num_layers=3,
                hidden_dims=(128, 64, 32),
                tau_mem_values=(5.0, 20.0, 30.0),
            ),
            SNNDeep,
        ),
        (
            ArchSpec(
                name="coincidence_gated",
                description="Left/right projections with multiplicative coincidence gating",
                num_layers=2,
                hidden_dims=(128, 64),
                tau_mem_values=(5.0, 20.0),
            ),
            SNNCoincidenceGated,
        ),
        (
            ArchSpec(
                name="parallel_mso",
                description="Parallel MSO subpops with fast+slow taus",
                num_layers=2,
                hidden_dims=(128, 64),
                tau_mem_values=(4.0, 7.0, 20.0),
            ),
            SNNParallelMSO,
        ),
        (
            ArchSpec(
                name="residual_readout",
                description="MSO->IC with residual from MSO to readout",
                num_layers=2,
                hidden_dims=(128, 64),
                tau_mem_values=(5.0, 20.0),
            ),
            SNNResidualReadout,
        ),
    ]


def _suggest_params(trial: optuna.Trial, arch_name: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params["lr"] = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    params["surrogate_slope"] = trial.suggest_float("surrogate_slope", 10.0, 50.0)

    params["hidden_mso"] = trial.suggest_categorical("hidden_mso", [96, 128])
    params["hidden_ic"] = trial.suggest_categorical("hidden_ic", [48, 64])

    params["tau_mso_ms"] = trial.suggest_float("tau_mso_ms", 3.0, 8.0)
    params["tau_ic_ms"] = trial.suggest_float("tau_ic_ms", 10.0, 30.0)

    params["thr_mso"] = trial.suggest_float("thr_mso", 0.8, 1.6)
    params["thr_ic"] = trial.suggest_float("thr_ic", 0.8, 1.4)

    if arch_name == "deep":
        params["hidden_hi"] = trial.suggest_categorical("hidden_hi", [24, 32, 48])
        params["tau_hi_ms"] = trial.suggest_float("tau_hi_ms", 20.0, 40.0)
        params["thr_hi"] = trial.suggest_float("thr_hi", 0.8, 1.4)

    if arch_name == "parallel_mso":
        params["tau_mso_fast_ms"] = trial.suggest_float("tau_mso_fast_ms", 3.0, 5.0)
        params["tau_mso_slow_ms"] = trial.suggest_float("tau_mso_slow_ms", 6.0, 9.0)

    return params


def _build_model(arch_name: str, model_cls: Any, params: Dict[str, Any], in_features: int) -> nn.Module:
    if arch_name == "baseline":
        return model_cls(
            in_features=in_features,
            hidden_mso=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            tau_mso_ms=params["tau_mso_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            surrogate_slope=params["surrogate_slope"],
        )
    if arch_name == "skip_ic":
        return model_cls(
            in_features=in_features,
            hidden_mso=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            tau_mso_ms=params["tau_mso_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            surrogate_slope=params["surrogate_slope"],
        )
    if arch_name == "deep":
        return model_cls(
            in_features=in_features,
            hidden_mso=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            hidden_hi=params["hidden_hi"],
            tau_mso_ms=params["tau_mso_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            tau_hi_ms=params["tau_hi_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            thr_hi=params["thr_hi"],
            surrogate_slope=params["surrogate_slope"],
        )
    if arch_name == "coincidence_gated":
        return model_cls(
            in_features=in_features,
            hidden_mso=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            tau_mso_ms=params["tau_mso_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            surrogate_slope=params["surrogate_slope"],
        )
    if arch_name == "parallel_mso":
        return model_cls(
            in_features=in_features,
            hidden_mso_total=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            tau_mso_fast_ms=params["tau_mso_fast_ms"],
            tau_mso_slow_ms=params["tau_mso_slow_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            surrogate_slope=params["surrogate_slope"],
        )
    if arch_name == "residual_readout":
        return model_cls(
            in_features=in_features,
            hidden_mso=params["hidden_mso"],
            hidden_ic=params["hidden_ic"],
            tau_mso_ms=params["tau_mso_ms"],
            tau_ic_ms=params["tau_ic_ms"],
            thr_mso=params["thr_mso"],
            thr_ic=params["thr_ic"],
            surrogate_slope=params["surrogate_slope"],
        )

    raise ValueError(f"Unknown architecture: {arch_name}")


def _train_one_trial(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    norm: TargetNorm,
    lr: float,
    max_epochs: int,
    grad_clip: float,
) -> Tuple[bool, Dict[str, Any]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = math.inf
    best_theta = math.inf
    best_range = math.inf
    best_state = None
    grad_status = "unknown"

    for _ in range(max_epochs):
        train_loss, _, _, grad_status = _run_epoch(
            model, train_loader, optimizer, device, True, norm, grad_clip
        )
        val_loss, val_theta, val_range, _ = _run_epoch(
            model, val_loader, optimizer, device, False, norm, grad_clip
        )

        if _detect_divergence(train_loss) or _detect_divergence(val_loss):
            return False, {"reason": "diverged"}

        if val_loss < best_val:
            best_val = val_loss
            best_theta = val_theta
            best_range = val_range
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        return False, {"reason": "no_best"}

    return True, {
        "best_val": best_val,
        "best_theta": best_theta,
        "best_range": best_range,
        "best_state": best_state,
        "grad_status": grad_status,
    }


def _run_attempt(
    attempt_number: int,
    phase: str,
    arch_spec: ArchSpec,
    model_cls: Any,
    config: SearchConfig,
    dataset: EchoDataset,
    train_subset: Subset,
    val_subset: Subset,
    device: torch.device,
    n_trials: int,
    max_epochs: int,
) -> Dict[str, Any]:
    best = {
        "val": math.inf,
        "theta": math.inf,
        "range": math.inf,
        "params": None,
        "state": None,
        "grad_status": "unknown",
    }

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, arch_spec.name)
        params["noise_std"] = config.noise_std

        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        model = _build_model(arch_spec.name, model_cls, params, in_features=config.n_channels * 2)
        model = model.to(device)

        success, stats = _train_one_trial(
            model,
            train_loader,
            val_loader,
            device,
            dataset.norm,
            params["lr"],
            max_epochs,
            config.grad_clip,
        )

        comments = ""
        if not success and stats.get("reason") == "diverged":
            comments = "restart after divergence: lr*0.5, thr*0.9"
            params["lr"] = max(params["lr"] * 0.5, 1e-4)
            params["thr_mso"] *= 0.9
            params["thr_ic"] *= 0.9

            model = _build_model(arch_spec.name, model_cls, params, in_features=config.n_channels * 2)
            model = model.to(device)

            success, stats = _train_one_trial(
                model,
                train_loader,
                val_loader,
                device,
                dataset.norm,
                params["lr"],
                max_epochs,
                config.grad_clip,
            )

        if not success:
            _append_trial_log(
                [attempt_number, phase, trial.number + 1, "inf", "inf", "inf", json.dumps(params), "diverged", comments]
            )
            return float("inf")

        if stats["best_val"] < best["val"]:
            best["val"] = stats["best_val"]
            best["theta"] = stats["best_theta"]
            best["range"] = stats["best_range"]
            best["params"] = params
            best["state"] = stats["best_state"]
            best["grad_status"] = stats["grad_status"]

        _append_trial_log(
            [
                attempt_number,
                phase,
                trial.number + 1,
                f"{stats['best_val']:.6f}",
                f"{stats['best_theta']:.3f}",
                f"{stats['best_range']:.3f}",
                json.dumps(params),
                stats["grad_status"],
                comments,
            ]
        )

        return stats["best_val"]

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config.seed + attempt_number),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Compute spike rates on validation set for best params
    if best["state"] is not None:
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        model = _build_model(arch_spec.name, model_cls, best["params"], in_features=config.n_channels * 2)
        model.load_state_dict(best["state"])
        model = model.to(device)
        spk1, spk2 = _eval_spike_rates(model, val_loader, device)
    else:
        spk1, spk2 = 0.0, 0.0

    return {
        "best": best,
        "spk1": spk1,
        "spk2": spk2,
    }


def _viability(theta_mae: float, range_mae: float) -> str:
    if theta_mae > 30.0 or range_mae > 10.0:
        return "N"
    if theta_mae < 15.0 and range_mae < 6.0:
        return "Y"
    return "N"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = SearchConfig()
    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)
    device = _select_device(config.use_mps_if_available)

    dataset = EchoDataset(
        n_samples=config.n_samples,
        seq_len_ms=config.seq_len_ms,
        n_channels=config.n_channels,
        seed=config.seed,
        cache=True,
        noise_std=config.noise_std,
    )

    n_val = int(config.n_samples * config.val_split)
    n_train = config.n_samples - n_val
    indices = torch.randperm(config.n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())

    _init_logs()

    arch_list = _build_arch_list()

    viable_archs: List[Tuple[ArchSpec, Any]] = []

    # Phase 1: attempts 1-6
    for i, (arch_spec, model_cls) in enumerate(arch_list, start=1):
        if i > 6:
            break
        result = _run_attempt(
            attempt_number=i,
            phase="Phase1",
            arch_spec=arch_spec,
            model_cls=model_cls,
            config=config,
            dataset=dataset,
            train_subset=train_subset,
            val_subset=val_subset,
            device=device,
            n_trials=config.phase1_trials,
            max_epochs=config.phase1_epochs,
        )

        best = result["best"]
        viable = _viability(best["theta"], best["range"])
        if viable == "Y":
            viable_archs.append((arch_spec, model_cls))

        _append_arch_log(
            [
                i,
                "Phase1",
                arch_spec.description,
                arch_spec.num_layers,
                str(arch_spec.hidden_dims),
                str(arch_spec.tau_mem_values),
                f"{result['spk1']:.4f}",
                f"{result['spk2']:.4f}",
                f"{best['theta']:.3f}",
                f"{best['range']:.3f}",
                viable,
                f"Grad: {best['grad_status']}",
            ]
        )

        # Early stop if target achieved
        if best["theta"] < 5.0 and best["range"] < 2.0:
            print("Objective reached in Phase 1. Stopping early.")
            return

    # Phase 2: attempts 7-10
    attempt_num = 7
    for arch_spec, model_cls in viable_archs:
        if attempt_num > 10:
            break
        result = _run_attempt(
            attempt_number=attempt_num,
            phase="Phase2",
            arch_spec=arch_spec,
            model_cls=model_cls,
            config=config,
            dataset=dataset,
            train_subset=train_subset,
            val_subset=val_subset,
            device=device,
            n_trials=config.phase2_trials,
            max_epochs=config.phase2_epochs,
        )

        best = result["best"]
        viable = "Y" if (best["theta"] < 15.0 and best["range"] < 6.0) else "N"

        _append_arch_log(
            [
                attempt_num,
                "Phase2",
                arch_spec.description,
                arch_spec.num_layers,
                str(arch_spec.hidden_dims),
                str(arch_spec.tau_mem_values),
                f"{result['spk1']:.4f}",
                f"{result['spk2']:.4f}",
                f"{best['theta']:.3f}",
                f"{best['range']:.3f}",
                viable,
                f"Grad: {best['grad_status']}",
            ]
        )

        if best["theta"] < 5.0 and best["range"] < 2.0:
            print("Objective reached in Phase 2. Stopping early.")
            return

        attempt_num += 1

    print("Search complete. See architecture_search_log.csv and architecture_trials_log.csv.")


if __name__ == "__main__":
    main()
