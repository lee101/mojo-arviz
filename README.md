# mojo-arviz

`mojo-arviz` is a standalone Mojo port of the compute-heavy posterior diagnostics in
[ArviZ](https://python.arviz.org/). It provides an array-oriented Python API with the same
public names and signatures as ArviZ 1.2.0 for the covered functions:

```python
import mojoarviz as az

az.ess(...)
az.rhat(...)
az.hdi(...)
```

The project is useful today for convergence checks on large posterior arrays. Its benchmark
compares each covered kernel with upstream on identical inputs and checks the results for
numerical agreement before timing.

## Coverage

- `ess`: `bulk`, `tail`, `mean`, `sd`, `median`, `mad`, `z_scale`, `folded`,
  `identity`, `quantile`, and `local`; absolute and relative ESS
- `rhat`: `rank`, `identity`, `split`, `z_scale`, and `folded`
- `hdi`: nearest-sample HDI, including arbitrary reduction axes, circular samples, and
  `skipna`
- NumPy-like arrays with configurable chain and draw axes
- xarray `DataArray` and `Dataset` inputs, including variable and coordinate selection

HDI `multimodal` and `multimodal_sample` are not implemented. The rest of ArviZ—including
MCSE, BFMI, PSIS/LOO, summaries, plotting, data conversion, Dask execution, and inference
backends—is outside this focused port.

Parity tests run against the real `arviz` package from conda-forge. They cover every ESS and
R-hat method listed above, HDI probabilities and axes, tied and autocorrelated draws, odd draw
counts, single chains, invalid/non-finite inputs, circular samples, and xarray containers.

The suite does not merely check that the port runs: each result is compared with the installed
ArviZ (or, for the numeric kernels, with `arviz_stats`' own `_z_scale`, `_ess`, `_rhat`, and
`_hdi_linear_nearest_common`) at a tolerance stated in the test. Degenerate inputs are pinned
too: zero-size arrays return `NaN` where ArviZ does and raise the same error family where it
raises, `inf` follows ArviZ's NaN-only validity check, duplicate draws match ArviZ's tie
handling, and DataArray/Dataset results keep upstream's dims, coordinates, names, and attrs.

## Install

Install [Pixi](https://pixi.sh/), then create the pinned environment and build the shared
library:

```bash
pixi install
pixi run build
```

The Python package is available through the Pixi environment's configured `PYTHONPATH`.
For packaging outside the checkout, build the library and set `MOJOARVIZ_LIB` to its absolute
path.

## Usage

This example creates four autocorrelated chains and computes the three covered diagnostics:

```python
import numpy as np
import mojoarviz as az

rng = np.random.default_rng(7)
noise = rng.normal(size=(4, 2_000))
posterior = np.empty_like(noise)
posterior[:, 0] = noise[:, 0]
for draw in range(1, posterior.shape[1]):
    posterior[:, draw] = 0.8 * posterior[:, draw - 1] + noise[:, draw]

print(az.ess(posterior))
print(az.rhat(posterior))
print(az.hdi(posterior, prob=0.89, axis=None))
```

The example is checked in as `examples/basic.py`; run it with
`pixi run python examples/basic.py`. The same functions accept extra parameter dimensions,
xarray `DataArray` objects, and xarray `Dataset` objects.

## How it works

Python normalizes axes and converts each sample block to C-contiguous row-major `float64`.
Buffers cross the C boundary as integer addresses; Mojo reconstructs
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values internally. All input, output, and scratch
memory remains owned by NumPy, so the shared library performs no cross-language allocation.
One compilation unit produces `dist/libmojo-arviz.so`.

The ESS kernel uses an in-place Mojo radix-two FFT for arrays up to 65,536 values and a
batched NumPy real FFT above that cutoff. Native FFT butterflies, range scans, setup,
scaling, power, and reduction loops use host-width float64 SIMD with scalar tails. Native
jobs with at least 131,072 padded chain values parallelize across independent chains; smaller
jobs remain serial. The inverse-FFT buffer is consumed directly by Geyer's
initial-positive and initial-monotone sequences, avoiding a separate autocovariance buffer
and copy.

Rank diagnostics use a Mojo quicksort, tie-aware average ranks, Blom's rank back-transform,
and a standalone inverse-normal approximation. Rank buffer initialization is SIMD
vectorized, and large rank R-hat jobs normalize the bulk and folded arrays concurrently.
Nearest HDI uses NumPy's optimized sort once and passes that buffer zero-copy to Mojo for
shortest-window selection.

    report. The R-hat reduction sums each chain and its squared deviations with two
    independent host-width SIMD vectors per chain instead of one scalar dependency chain,
    so it streams at memory bandwidth rather than add latency. The wider accumulation
    changes the addition order, so R-hat results can differ from a strictly sequential sum
    in the last bits.

No GPU path is included. The FFT butterfly performs roughly 10 floating-point operations
while moving at least 32 bytes per stage, below the requested two-flops-per-byte threshold;
the other kernels are sorting, bandwidth, or branch dominated. Host/device transfers and
launches would therefore make these paths slower rather than useful.

## Correctness and tests

```bash
pixi run build
pixi run test
```

The parity suite exercises every method listed in Coverage against ArviZ and includes
non-contiguous axes, SIMD tails, threshold boundaries, invalid inputs, circular intervals,
and xarray containers.

## Benchmarks

Measured with `pixi run bench`; each result is the best of five warmed runs. A ratio above
1 means `mojoarviz` is faster. "Before" is the same command on the same machine before the
reduction rewrite below; both sessions were on a shared host, so treat the two-decimal
figures as comparable to about ±10 %.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.
Versions: mojoarviz 0.1.0; ArViZ 1.2.0; NumPy 2.5.1.

| case | before | after | ArviZ | ArViZ / Mojo | result |
|---|---:|---:|---:|---:|---|
| R-hat identity (4 x 1,000,000) | 22.04 ms | 7.35 ms | 41.76 ms | 5.68x | faster |
| R-hat rank (4 x 100,000) | 113.07 ms | 114.00 ms | 169.33 ms | 1.49x | faster |
| ESS identity AR(1) (4 x 16,384) | 2.87 ms | 2.70 ms | 5.02 ms | 1.86x | faster |
| ESS bulk AR(1) (4 x 16,384) | 10.60 ms | 9.72 ms | 15.70 ms | 1.62x | faster |
| HDI nearest (1,000,000) | 19.29 ms | 19.01 ms | 21.46 ms | 1.13x | faster |

The R-hat identity row is the one this rewrite targets: the `mav_rhat` kernel alone, called
directly, went from 17.32 ms to 3.52 ms on 4 x 1,000,000 draws (4.9x), and from 1.58 ms to
0.31 ms on 4 x 100,000. The remaining rows are unchanged within host noise; the rank row
is dominated by the quicksort, which this phase measured and left alone (see below).

### SIMD phase (2026-11-11)

Measured in the same session, with the old and new shared libraries loaded side by side in
one Python process and called alternately for 21 rounds, so host drift hits both equally.
`min` and `median` are over those rounds. The host was carrying a load average of about 22
on 72 cores during the run, so the `pixi run bench` wall times below move by tens of
percent between runs; the interleaved kernel pairs are the stable comparison.

`mav_rhat` (the identity R-hat kernel), called directly:

| size | before min / median | after min / median | min ratio |
|---|---:|---:|---:|
| 4 x 1,000,000 | 5.02 / 5.62 ms | 3.57 / 3.81 ms | 1.41x |
| 4 x 300,000 | 1.68 / 1.81 ms | 0.95 / 1.04 ms | 1.77x |
| 4 x 100,000 | 0.54 / 0.57 ms | 0.32 / 0.34 ms | 1.72x |
| 4 x 16,384 | 0.09 / 0.10 ms | 0.05 / 0.05 ms | 1.74x |

End-to-end `pixi run bench`, three consecutive runs each on the old and the new library
(ArviZ / Mojo ratio), with the host load at ~22:

| case | before | after |
|---|---|---|
| R-hat identity (4 x 1,000,000) | 8.59, 9.23, 8.62 ms | 8.97, 10.53, 13.34 ms |
| R-hat rank (4 x 100,000) | 111.30, 111.89, 110.64 ms | 112.13, 114.18, 111.90 ms |
| ESS identity AR(1) (4 x 16,384) | 2.77, 2.73, 2.95 ms | 2.94, 3.03, 2.95 ms |
| ESS bulk AR(1) (4 x 16,384) | 10.86, 9.92, 10.31 ms | 9.87, 9.92, 10.14 ms |
| HDI nearest (1,000,000) | 18.15, 19.85, 23.24 ms | 19.52, 23.43, 20.41 ms |

The `R-hat identity` wall times do **not** show the kernel's 1.4–1.8x: on a host this loaded,
the numpy `isfinite` validity scan and the ctypes/Python wrapper, not the 4 ms kernel, set
that number. `mav_ess` was checked the same interleaved way and is unchanged (4 x 16,384:
2.70 vs 2.82 ms minimum; identical values), as is every other row.

What changed in `mav_rhat`: each chain is summed, then squared deviations are accumulated
against that chain's own mean, both with two independent host-width `SIMD` vectors instead
of eight scalar accumulators. The pass structure is deliberately unchanged, because the
faster fused version is not safe:

- **Two-pass pooled sum of squares, reverted.** One flat pass for `sum x^2` and one for the
  chain means gives `sum (x - xbar)^2 = sum x^2 - n * xbar^2`. That was 2.1x faster than
  the old kernel (4.45 -> 2.15 ms at 4 x 1,000,000), but the subtraction cancels: for a
  constant chain, and for the `[[1,1,1,1,2,2],[1,1,2,2,2,2]]` case the suite pins, it
  produced an invalid or negative within-chain variance where ArviZ returns 1.3333. It was
  removed, not tuned.
- **Tried and not kept, each measured on this host:** a streaming `sum x^2` pass over the
  whole flat buffer (no faster than per-chain sums); revisiting the quicksort, which is a
  sort-algorithm change rather than SIMD and belongs to its own phase. The existing
  single-vector `sum_values` helper (used by ESS) is in fact 26 % *slower* than the old
  eight-scalar `chain_sum` on one 32 MB buffer (1.69 vs 1.35 ms minimum over 31 interleaved
  rounds), which is why the rhat mean pass uses the two-vector pitch and not
  `sum_values`.
- **Rank kernel left alone.** A stage probe at 400,000 values measured prepare 0.60 ms,
  quicksort 43.8 ms, tie loop 3.85 ms; the tie loop's scatter is `dest[indices[i]]`, one
  gather per slot, so SIMD lanes cannot help it (8-wide store measured 34.6 vs 35.6 ms at
  best). Rank prepare was already vectorised before this phase.

The SIMD tail cases for both kernels are parametrised on the compiled `W`: lengths 0, 1,
W-1, W, W+1, 2W+3 and 200,003, plus unaligned chain starts (200,001 / 200,002 / 200,003 and
1,000,001 draws in a 3-chain buffer). The whole suite also passes with
`MOJOARVIZ_SIMD_WIDTH=8` and `=16`, which widens the tested length list beyond this host's
`W = 4`; that override changes the test lengths, not the compiled kernel, so it is the
scalar tail at a wider block that gets exercised. A deliberately broken scalar tail was
confirmed to fail the same tests.


### Summation order

`mav_rhat` sums each chain with two interleaved host-width SIMD vectors rather than one
running sum, and folds `1/(draws-1)` and `1/chains` into one division. Both change the order
of floating-point additions, so **results can differ from earlier releases, or from ArviZ,
in the last bits**. Measured against a 128-bit reference at 4 x 1,000,000, 4 x 100,000, 4 x
16,384, 2 x 7 and 5 x 13 draws, the vector accumulation is within 1e-16 of the exact value,
the same order of error as the earlier scalar and eight-accumulator loops. The parity
tolerances are untouched.

Three things were tried and **reverted** because `pixi run bench` did not show them faster:
bounds-check-free loads/stores in the quicksort (400,000-element sort 49–53 ms either way),
a two-accumulator Geyer `tau` sum (no change), and bounds-check-free loads in the HDI
window scan (77 us to 88 us at 1,000,000 values, worse).


To reproduce:

```bash
pixi run bench
```
