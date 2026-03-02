"""Resume architecture search from existing logs.

This script continues attempts that were not completed in
architecture_search_log.csv. It uses the same dataset split for
comparability (same seed and configuration).
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import torch

import architecture_search as search


def _read_completed_attempts(log_path: str) -> List[int]:
    if not os.path.exists(log_path):
        return []
    attempts = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                attempts.append(int(row["Attempt_Number"]))
            except (KeyError, ValueError):
                continue
    return sorted(set(attempts))


def main() -> None:
    config = search.SearchConfig()
    search.set_seed(config.seed)
    device = search._select_device(config.use_mps_if_available)

    dataset = search.EchoDataset(
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

    train_subset = search.Subset(dataset, train_idx.tolist())
    val_subset = search.Subset(dataset, val_idx.tolist())

    search._init_logs()

    completed = _read_completed_attempts(search.ARCH_LOG)

    arch_list = search._build_arch_list()

    # Phase 1: attempts 1-6
    viable_archs: List[Tuple[search.ArchSpec, object]] = []

    for i, (arch_spec, model_cls) in enumerate(arch_list, start=1):
        if i > 6:
            break
        if i in completed:
            # Determine viability from existing logs
            continue

        result = search._run_attempt(
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
        viable = search._viability(best["theta"], best["range"])
        if viable == "Y":
            viable_archs.append((arch_spec, model_cls))

        search._append_arch_log(
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

        if best["theta"] < 5.0 and best["range"] < 2.0:
            print("Objective reached in Phase 1. Stopping early.")
            return

    # Phase 2: use viable architectures from newly completed Phase 1 only
    attempt_num = 7
    for arch_spec, model_cls in viable_archs:
        if attempt_num > 10:
            break
        if attempt_num in completed:
            attempt_num += 1
            continue

        result = search._run_attempt(
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

        search._append_arch_log(
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

    print("Resume complete. See architecture_search_log.csv and architecture_trials_log.csv.")


if __name__ == "__main__":
    main()
