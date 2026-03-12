# Stage 1 Call Validation Results

This document summarizes the three call-generator variants tested in Stage 1 and their 5 m echo-delay validation outputs.

## Validation Metric Definitions

- `expected_delay_s`: theoretical round-trip delay, `2 * distance / c` (s)
- `measured_delay_s`: delay estimated by cross-correlation (s)
- `timing_error_s`: `measured_delay_s - expected_delay_s` (s)
- `range_error_m`: `0.5 * c * timing_error_s` (m)
- `sample_error`: `timing_error_s * fs` (samples)

## Results (Distance = 5.000 m)

| Call variant | fs (Hz) | expected_delay_s | measured_delay_s | timing_error_s | range_error_m | sample_error |
|---|---:|---:|---:|---:|---:|---:|
| Initial chirp baseline (pre FM-CF-FM refactor) | 256000 | 0.029154519 | 0.029156250 | 1.731049563e-06 | 2.968750000e-04 | 0.443149 |
| Active FM-CF-FM dual-harmonic (intermediate) | 12800 | 0.029154519 | 0.004765625 | -2.438889395e-02 | -4.182695312e+00 | -312.177843 |
| Hyperbolic FM-QCF (current) | 12800 | 0.029154519 | 0.029140625 | -1.389395044e-05 | -2.382812500e-03 | -0.177843 |

## Notes

- The intermediate active FM-CF-FM variant failed delay localization under the unchanged validation routine.
- The current Hyperbolic FM-QCF variant restores accurate delay estimation and is within ~1 sample at `fs=12800 Hz`.
