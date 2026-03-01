"""Physics-based binaural echo simulation for a bat-like CF-FM call.

All time axes are in milliseconds (ms) with 1 ms resolution to keep
compute and memory low. This is a deliberately simplified, coarse
approximation that is stable for SNN training.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Speed of sound in air (m/s)
SPEED_OF_SOUND = 343.0


def _linspace(start: float, end: float, steps: int) -> torch.Tensor:
    """Torch linspace helper that returns float32 on CPU."""
    return torch.linspace(start, end, steps, dtype=torch.float32)


def generate_cf_fm_call(
    cf_ms: int = 8,
    fm_ms: int = 4,
    dt_ms: int = 1,
    cf_freq_hz: float = 120.0,
    fm_start_hz: float = 180.0,
    fm_end_hz: float = 60.0,
) -> torch.Tensor:
    """Generate a simplified CF-FM bat-like call.

    The sampling resolution is 1 ms (1 kHz). This is too low for real
    ultrasonic bat calls, so we intentionally use baseband frequencies
    to create a stable, trainable surrogate waveform.

    Returns
    -------
    call : torch.Tensor
        Shape [T_call], amplitude in [-1, 1].
    """
    assert dt_ms == 1, "This project assumes 1 ms time resolution."

    # CF segment
    t_cf = _linspace(0.0, (cf_ms - 1) * 1e-3, cf_ms)
    cf_wave = torch.sin(2.0 * math.pi * cf_freq_hz * t_cf)

    # FM segment (linear chirp)
    t_fm = _linspace(0.0, (fm_ms - 1) * 1e-3, fm_ms)
    # Frequency sweep from fm_start_hz to fm_end_hz over fm_ms
    f_t = _linspace(fm_start_hz, fm_end_hz, fm_ms)
    # Phase is integral of frequency
    phase = 2.0 * math.pi * torch.cumsum(f_t, dim=0) * 1e-3
    fm_wave = torch.sin(phase)

    call = torch.cat([cf_wave, fm_wave], dim=0)
    # Light tapering to avoid hard edges
    window = torch.hann_window(call.numel(), periodic=False)
    call = call * window
    return call


def simulate_binaural_echo(
    theta_deg: float,
    r_m: float,
    seq_len_ms: int = 200,
    dt_ms: int = 1,
    noise_std: float = 0.01,
    ear_separation_m: float = 0.02,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate left/right ear waveforms for a single point reflector.

    Geometry:
        - Bat at origin (0, 0)
        - Target at (r, theta)
        - Left ear at (-d/2, 0), right ear at (+d/2, 0)

    Echo model:
        - Round-trip delay from bat to target, plus target to ear
        - Spherical spreading attenuation ~ 1 / r^2
        - Additive Gaussian noise

    Returns
    -------
    left, right : torch.Tensor
        Each is shape [T], sampled at 1 ms resolution.
    """
    assert dt_ms == 1, "This project assumes 1 ms time resolution."

    # Convert polar to Cartesian for target
    theta_rad = math.radians(theta_deg)
    x = r_m * math.cos(theta_rad)
    y = r_m * math.sin(theta_rad)

    # Ear positions
    left_ear = (-ear_separation_m / 2.0, 0.0)
    right_ear = (ear_separation_m / 2.0, 0.0)

    # Distance from bat (origin) to target
    dist_emit = r_m

    # Distance from target to each ear
    dist_left = math.sqrt((x - left_ear[0]) ** 2 + (y - left_ear[1]) ** 2)
    dist_right = math.sqrt((x - right_ear[0]) ** 2 + (y - right_ear[1]) ** 2)

    # Total path length (emit + receive)
    total_left = dist_emit + dist_left
    total_right = dist_emit + dist_right

    # Convert to delay in ms
    delay_left_ms = int(round((total_left / speed_of_sound) * 1000.0))
    delay_right_ms = int(round((total_right / speed_of_sound) * 1000.0))

    # Generate call
    call = generate_cf_fm_call()
    call_len = call.numel()

    # Initialize waveforms
    left = torch.zeros(seq_len_ms, dtype=torch.float32)
    right = torch.zeros(seq_len_ms, dtype=torch.float32)

    # Simple spherical spreading attenuation
    attenuation = 1.0 / max(r_m ** 2, 1e-6)

    def _add_echo(wave: torch.Tensor, delay_ms: int) -> None:
        if delay_ms >= seq_len_ms:
            return
        end = min(seq_len_ms, delay_ms + call_len)
        length = end - delay_ms
        if length > 0:
            wave[delay_ms:end] += attenuation * call[:length]

    _add_echo(left, delay_left_ms)
    _add_echo(right, delay_right_ms)

    # Additive Gaussian noise
    if noise_std > 0.0:
        left += noise_std * torch.randn_like(left)
        right += noise_std * torch.randn_like(right)

    return left, right


if __name__ == "__main__":
    # Quick sanity check
    l, r = simulate_binaural_echo(theta_deg=30.0, r_m=10.0)
    print("Left/Right shapes:", l.shape, r.shape)
    print("Left max/min:", float(l.max()), float(l.min()))
