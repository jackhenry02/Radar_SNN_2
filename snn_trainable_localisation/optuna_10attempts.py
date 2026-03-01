"""Run 10 Optuna trials with 10-epoch cap and log results.

This script creates a trailing log of all attempts with parameters and
MAE metrics, including a comments column for any special notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Optuna is not installed. Install it with: pip install optuna"
    ) from exc

from snn_model import SNNLocalizer
from utils import EchoDataset, TargetNorm, compute_errors, denormalize_targets, set_seed
from plots import plot_predictions


@dataclass
class TuneConfig:
    n_samples: int = 1000
    seq_len_ms: int = 200
    n_channels: int = 32
    batch_size: int = 16
    val_split: float = 0.2
    noise_std: float = 0.005

    max_epochs: int = 10
    patience: int = 5
    grad_clip: float = 1.0

    use_mps_if_available: bool = True
    n_trials: int = 10
    seed: int = 42


LOG_PATH = os.path.join(THIS_DIR, "trial_log.csv")


def _select_device(use_mps: bool) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_time_major(spikes: torch.Tensor) -> torch.Tensor:
    return spikes.permute(1, 0, 2).contiguous()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    norm: TargetNorm,
    grad_clip: float,
) -> Tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_theta = 0.0
    total_range = 0.0
    count = 0

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
            optimizer.step()

        theta_err, range_err = compute_errors(outputs.detach(), targets, norm)

        total_loss += float(loss.item()) * spikes.size(1)
        total_theta += float(theta_err.mean().item()) * spikes.size(1)
        total_range += float(range_err.mean().item()) * spikes.size(1)
        count += spikes.size(1)

    if count == 0:
        return math.inf, math.inf, math.inf

    return total_loss / count, total_theta / count, total_range / count


def _detect_divergence(loss: float) -> bool:
    if math.isnan(loss) or math.isinf(loss):
        return True
    if loss > 1e3:
        return True
    return False


def _train_with_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    norm: TargetNorm,
    lr: float,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    trial: optuna.Trial | None,
) -> Tuple[bool, Dict[str, Any]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = math.inf
    best_theta = math.inf
    best_range = math.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, _, _ = _run_epoch(
            model, train_loader, optimizer, device, True, norm, grad_clip
        )
        val_loss, val_theta, val_range = _run_epoch(
            model, val_loader, optimizer, device, False, norm, grad_clip
        )

        if _detect_divergence(train_loss) or _detect_divergence(val_loss):
            return False, {"reason": "diverged"}

        if val_loss < best_val:
            best_val = val_loss
            best_theta = val_theta
            best_range = val_range
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if epochs_no_improve >= patience:
            break

    if best_state is None:
        return False, {"reason": "no_best"}

    return True, {
        "best_val": best_val,
        "best_theta": best_theta,
        "best_range": best_range,
        "best_state": best_state,
    }


def _init_log(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "attempt",
                "val_loss",
                "theta_mae_deg",
                "range_mae_m",
                "params_json",
                "comments",
            ]
        )


def _append_log(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = TuneConfig()
    if args.trials is not None:
        config.n_trials = args.trials
    if args.epochs is not None:
        config.max_epochs = args.epochs
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

    _init_log(LOG_PATH)

    best_global = {
        "loss": math.inf,
        "state": None,
        "params": None,
        "theta": None,
        "range": None,
    }

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        tau_mso = trial.suggest_float("tau_mso_ms", 3.0, 8.0)
        tau_ic = trial.suggest_float("tau_ic_ms", 10.0, 30.0)
        thr_mso = trial.suggest_float("thr_mso", 0.8, 1.6)
        thr_ic = trial.suggest_float("thr_ic", 0.8, 1.4)
        hidden_mso = trial.suggest_categorical("hidden_mso", [96, 128])
        hidden_ic = trial.suggest_categorical("hidden_ic", [48, 64])

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

        model = SNNLocalizer(
            in_features=config.n_channels * 2,
            hidden_mso=hidden_mso,
            hidden_ic=hidden_ic,
            tau_mso_ms=tau_mso,
            tau_ic_ms=tau_ic,
            thr_mso=thr_mso,
            thr_ic=thr_ic,
        ).to(device)

        success, stats = _train_with_config(
            model,
            train_loader,
            val_loader,
            device,
            dataset.norm,
            lr,
            config.max_epochs,
            config.patience,
            config.grad_clip,
            trial,
        )

        comments = ""
        # One restart on divergence (per rules)
        if not success and stats.get("reason") == "diverged":
            comments = "restart after divergence: lr*0.5, thr*0.9"
            lr = max(lr * 0.5, 1e-4)
            thr_mso = thr_mso * 0.9
            thr_ic = thr_ic * 0.9

            model = SNNLocalizer(
                in_features=config.n_channels * 2,
                hidden_mso=hidden_mso,
                hidden_ic=hidden_ic,
                tau_mso_ms=tau_mso,
                tau_ic_ms=tau_ic,
                thr_mso=thr_mso,
                thr_ic=thr_ic,
            ).to(device)

            success, stats = _train_with_config(
                model,
                train_loader,
                val_loader,
                device,
                dataset.norm,
                lr,
                config.max_epochs,
                config.patience,
                config.grad_clip,
                trial,
            )

        if not success:
            params = {
                "lr": lr,
                "tau_mso_ms": tau_mso,
                "tau_ic_ms": tau_ic,
                "thr_mso": thr_mso,
                "thr_ic": thr_ic,
                "hidden_mso": hidden_mso,
                "hidden_ic": hidden_ic,
                "noise_std": config.noise_std,
            }
            _append_log(
                LOG_PATH,
                [trial.number + 1, "inf", "inf", "inf", json.dumps(params), comments],
            )
            return float("inf")

        if stats["best_val"] < best_global["loss"]:
            best_global["loss"] = stats["best_val"]
            best_global["state"] = stats["best_state"]
            best_global["theta"] = stats["best_theta"]
            best_global["range"] = stats["best_range"]
            best_global["params"] = {
                "lr": lr,
                "tau_mso_ms": tau_mso,
                "tau_ic_ms": tau_ic,
                "thr_mso": thr_mso,
                "thr_ic": thr_ic,
                "hidden_mso": hidden_mso,
                "hidden_ic": hidden_ic,
                "noise_std": config.noise_std,
            }

        params = {
            "lr": lr,
            "tau_mso_ms": tau_mso,
            "tau_ic_ms": tau_ic,
            "thr_mso": thr_mso,
            "thr_ic": thr_ic,
            "hidden_mso": hidden_mso,
            "hidden_ic": hidden_ic,
            "noise_std": config.noise_std,
        }
        _append_log(
            LOG_PATH,
            [
                trial.number + 1,
                f"{stats['best_val']:.6f}",
                f"{stats['best_theta']:.3f}",
                f"{stats['best_range']:.3f}",
                json.dumps(params),
                comments,
            ],
        )

        return stats["best_val"]

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=config.seed),
        pruner=MedianPruner(n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=True)

    # Save best model and plots for the best trial
    if best_global["state"] is not None:
        model_path = os.path.join(THIS_DIR, "snn_localizer_optuna_10ep.pt")
        torch.save(best_global["state"], model_path)

        best_params = best_global["params"]
        model = SNNLocalizer(
            in_features=config.n_channels * 2,
            hidden_mso=best_params["hidden_mso"],
            hidden_ic=best_params["hidden_ic"],
            tau_mso_ms=best_params["tau_mso_ms"],
            tau_ic_ms=best_params["tau_ic_ms"],
            thr_mso=best_params["thr_mso"],
            thr_ic=best_params["thr_ic"],
        ).to(device)
        model.load_state_dict(best_global["state"])
        model.eval()

        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        theta_true: List[float] = []
        theta_pred: List[float] = []
        r_true: List[float] = []
        r_pred: List[float] = []

        with torch.no_grad():
            for spikes, targets in val_loader:
                spikes = spikes.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                outputs, _ = model(_batch_to_time_major(spikes))

                outputs_denorm = denormalize_targets(outputs.cpu(), dataset.norm)
                targets_denorm = denormalize_targets(targets.cpu(), dataset.norm)

                theta_true.extend(targets_denorm[:, 0].tolist())
                theta_pred.extend(outputs_denorm[:, 0].tolist())
                r_true.extend(targets_denorm[:, 1].tolist())
                r_pred.extend(outputs_denorm[:, 1].tolist())

        plot_predictions(theta_true, theta_pred, r_true, r_pred, THIS_DIR)

        print(f"Saved best model to: {model_path}")
        print(f"Trial log saved to: {LOG_PATH}")
    else:
        print("No successful trials to save.")


if __name__ == "__main__":
    main()
