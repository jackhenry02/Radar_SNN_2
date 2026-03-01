"""Train the SNN localizer on simulated binaural echoes.

Run:
    python snn_trainable_localisation/train.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from snn_model import SNNLocalizer
from utils import (
    EchoDataset,
    TargetNorm,
    compute_errors,
    denormalize_targets,
    set_seed,
)
from plots import plot_predictions


@dataclass
class TrainConfig:
    # Data
    n_samples: int = 1000
    seq_len_ms: int = 200
    n_channels: int = 32
    batch_size: int = 16
    val_split: float = 0.2
    noise_std: float = 0.01

    # Model
    hidden_mso: int = 128
    hidden_ic: int = 64
    tau_mso_ms: float = 5.0
    tau_ic_ms: float = 20.0
    thr_mso: float = 1.2
    thr_ic: float = 1.0

    # Training
    lr: float = 5e-4
    max_epochs: int = 50
    grad_clip: float = 1.0
    patience: int = 10

    # Misc
    seed: int = 42
    use_mps_if_available: bool = True


def _select_device(use_mps: bool) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_time_major(spikes: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, C] to [T, B, C]."""
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
    """Run one epoch and return (loss, theta_err, range_err)."""
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


def _write_experiment_log(
    out_path: str,
    config: TrainConfig,
    best_val: float,
    best_theta: float,
    best_range: float,
    notes: str,
) -> None:
    """Write experiment notes to a text file."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SNN Trainable Localisation Experiment Log\n")
        f.write("=" * 48 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Samples: {config.n_samples}\n")
        f.write(f"  Sequence length (ms): {config.seq_len_ms}\n")
        f.write(f"  Channels: {config.n_channels}\n")
        f.write(f"  Batch size: {config.batch_size}\n")
        f.write(f"  LR: {config.lr}\n")
        f.write(f"  Epochs (max): {config.max_epochs}\n\n")

        f.write("Best Validation Metrics:\n")
        f.write(f"  Val loss: {best_val:.6f}\n")
        f.write(f"  Theta MAE (deg): {best_theta:.3f}\n")
        f.write(f"  Range MAE (m): {best_range:.3f}\n\n")

        f.write("Notes:\n")
        f.write(notes.strip() + "\n")


def train_once(config: TrainConfig, attempt: int = 1) -> Tuple[bool, dict]:
    """Train for one run. Returns (success, stats)."""
    set_seed(config.seed + attempt)
    device = _select_device(config.use_mps_if_available)

    # Dataset + loaders
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

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = SNNLocalizer(
        in_features=config.n_channels * 2,
        hidden_mso=config.hidden_mso,
        hidden_ic=config.hidden_ic,
        tau_mso_ms=config.tau_mso_ms,
        tau_ic_ms=config.tau_ic_ms,
        thr_mso=config.thr_mso,
        thr_ic=config.thr_ic,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val = math.inf
    best_theta = math.inf
    best_range = math.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_theta, train_range = _run_epoch(
            model, train_loader, optimizer, device, True, dataset.norm, config.grad_clip
        )
        val_loss, val_theta, val_range = _run_epoch(
            model, val_loader, optimizer, device, False, dataset.norm, config.grad_clip
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.6f} | Val loss {val_loss:.6f} | "
            f"Theta MAE {val_theta:.2f} deg | Range MAE {val_range:.2f} m"
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

        if epochs_no_improve >= config.patience:
            print("Early stopping: no improvement.")
            break

    if best_state is None:
        return False, {"reason": "no_best"}

    # Save model
    model_path = os.path.join("snn_trainable_localisation", "snn_localizer.pt")
    torch.save(best_state, model_path)

    # Evaluate on validation set for plotting
    model.load_state_dict(best_state)
    model.eval()

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

    plot_predictions(theta_true, theta_pred, r_true, r_pred, "snn_trainable_localisation")

    stats = {
        "best_val": best_val,
        "best_theta": best_theta,
        "best_range": best_range,
        "model_path": model_path,
    }
    return True, stats


def main() -> None:
    config = TrainConfig()

    # Attempt training, with one allowed restart on divergence
    success, stats = train_once(config, attempt=1)

    if not success and stats.get("reason") == "diverged":
        print("Divergence detected: lowering LR and thresholds, restarting once...")
        config.lr = max(config.lr * 0.5, 1e-4)
        config.thr_mso *= 0.9
        config.thr_ic *= 0.9
        success, stats = train_once(config, attempt=2)

    if not success:
        print("Training did not complete successfully.")
        return

    # Write experiment notes
    notes = """
Architecture reasoning:
- Two-layer LIF network mimics MSO (coincidence detection) and IC (temporal integration).
- Short tau in MSO emphasizes precise timing; longer tau in IC integrates over echo duration.

Observed convergence behaviour:
- Fill in after running training. The script logs losses per epoch and saves best validation metrics.

Strengths:
- Biologically inspired timing sensitivity with a compact, trainable architecture.
- Efficient for low-resolution time steps and limited channel count.

Weaknesses:
- Coarse temporal resolution (1 ms) limits fine ITD cues.
- Simplified filterbank and call model may underrepresent real bat acoustics.

Biological plausibility comments:
- LIF neurons and layered structure are plausible, but synaptic dynamics and cochlear
  mechanics are simplified for stability and speed.
"""

    log_path = os.path.join("snn_trainable_localisation", "experiment_log.txt")
    _write_experiment_log(
        log_path,
        config,
        stats["best_val"],
        stats["best_theta"],
        stats["best_range"],
        notes,
    )

    print(f"Saved model to: {stats['model_path']}")
    print(f"Wrote experiment log to: {log_path}")


if __name__ == "__main__":
    main()
