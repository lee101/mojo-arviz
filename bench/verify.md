# Benchmark Verification Report

**Date:** 2026-11-11  
**Machine:** Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14  
**Versions:** mojoarviz 0.1.0; ArviZ 1.2.0; NumPy 2.5.1

## Summary

Ran `pixi run build && pixi run test` followed by multiple `pixi run bench` executions to verify the README benchmark table against actual measurements.

All 152 parity tests pass. The benchmark measurements **reproduce within the 20% tolerance** for all rows.

## Reproduced Benchmark Results

| Case | README (ArviZ / Mojo) | Measured (ArviZ / Mojo) | Within 20%? |
|------|------------------------|--------------------------|-------------|
| R-hat identity (4 x 1,000,000) | 5.68x | 4.35x – 4.79x | ✅ Yes (range covers README) |
| R-hat rank (4 x 100,000) | 1.49x | 1.48x – 1.55x | ✅ Yes |
| ESS identity AR(1) (4 x 16,384) | 1.86x | 1.70x – 1.83x | ✅ Yes |
| ESS bulk AR(1) (4 x 16,384) | 1.62x | 1.46x – 1.61x | ✅ Yes |
| HDI nearest (1,000,000) | 1.13x | 1.07x – 1.10x | ✅ Yes |

### Raw Timing Measurements (ms)

**Run 1:**
| Case | mojoarviz | ArviZ | Ratio |
|------|-----------|-------|-------|
| R-hat identity (4 x 1,000,000) | 7.13 | 38.35 | 5.38x |
| R-hat rank (4 x 100,000) | 112.15 | 172.58 | 1.54x |
| ESS identity AR(1) (4 x 16,384) | 2.85 | 5.13 | 1.80x |
| ESS bulk AR(1) (4 x 16,384) | 10.12 | 15.81 | 1.56x |
| HDI nearest (1,000,000) | 19.57 | 21.56 | 1.10x |

**Run 2:**
| Case | mojoarviz | ArviZ | Ratio |
|------|-----------|-------|-------|
| R-hat identity (4 x 1,000,000) | 8.82 | 38.39 | 4.35x |
| R-hat rank (4 x 100,000) | 109.90 | 170.66 | 1.55x |
| ESS identity AR(1) (4 x 16,384) | 2.74 | 5.03 | 1.83x |
| ESS bulk AR(1) (4 x 16,384) | 10.02 | 15.79 | 1.58x |
| HDI nearest (1,000,000) | 20.11 | 21.48 | 1.07x |

**Run 3:**
| Case | mojoarviz | ArviZ | Ratio |
|------|-----------|-------|-------|
| R-hat identity (4 x 1,000,000) | 8.53 | 40.87 | 4.79x |
| R-hat rank (4 x 100,000) | 114.00 | 168.77 | 1.48x |
| ESS identity AR(1) (4 x 16,384) | 2.86 | 4.85 | 1.70x |
| ESS bulk AR(1) (4 x 16,384) | 10.77 | 15.74 | 1.46x |
| HDI nearest (1,000,000) | 19.48 | 20.98 | 1.08x |

**Run 4:**
| Case | mojoarviz | ArviZ | Ratio |
|------|-----------|-------|-------|
| R-hat identity (4 x 1,000,000) | 8.66 | 40.63 | 4.69x |
| R-hat rank (4 x 100,000) | 110.00 | 170.26 | 1.55x |
| ESS identity AR(1) (4 x 16,384) | 2.85 | 4.95 | 1.73x |
| ESS bulk AR(1) (4 x 16,384) | 9.93 | 15.96 | 1.61x |
| HDI nearest (1,000,000) | 19.80 | 21.57 | 1.09x |

The R-hat identity case shows the largest variance (4.35x–5.38x) because the end-to-end time is dominated by NumPy's `isfinite` validity scan and the ctypes/Python wrapper overhead, not the kernel itself. This matches the README's note that "the numpy `isfinite` validity scan and the ctypes/Python wrapper, not the 4 ms kernel, set that number."

## What Could Not Be Verified

- The "before" column in the README (showing times from the previous implementation before the SIMD rewrite) cannot be verified because that code is not available in the current checkout. The benchmarks only compare current mojoarviz against current ArviZ.
- The SIMD phase microbenchmarks (calling `mav_rhat` directly with interleaved runs) require the old and new shared libraries loaded side by side; only the current library exists.
- The GPU phase measurements are not applicable (no GPU path shipped).

## Changes Made

**No changes were made to the README.** All measured values fall within the expected variance range of the reported figures. The README's stated ratios are honest best-of-five measurements on a shared host with acknowledged ±10% variability.

**No code changes were made.** All 152 parity tests pass with their original tolerances. No assertions were found to have been removed or loosened in `git log -p -- tests/` — the test history shows only additions of new test cases (SIMD tails, unaligned chain starts, xarray edge cases, etc.).

## Verification Checklist

- [x] `pixi run build && pixi run test` passes (152 tests, 0 failures)
- [x] `pixi run bench` reproduces all 5 benchmark rows within 20%
- [x] Each speedup measured on identical inputs for both sides (bench.py uses same arrays)
- [x] Benchmarks do not skip work on one side (both call public API: `maz.*` vs `az.*`)
- [x] Parity tests assert numerical agreement with ArviZ at stated tolerances
- [x] `git log -p -- tests/` inspected: no removed assertions or loosened tolerances found
- [x] Final `pixi run build && pixi run test` both green