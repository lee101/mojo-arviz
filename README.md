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
1 means `mojoarviz` is faster.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.
Versions: mojoarviz 0.1.0; ArViZ 1.2.0; NumPy 2.5.1.

| case | mojoarviz | ArviZ | ArViZ / Mojo | result |
|---|---:|---:|---:|---|
| R-hat identity (4 x 1,000,000) | 18.78 ms | 152.62 ms | 8.13x | faster |
| R-hat rank (4 x 100,000) | 111.18 ms | 171.97 ms | 1.55x | faster |
| ESS identity AR(1) (4 x 16,384) | 2.90 ms | 5.19 ms | 1.79x | faster |
| ESS bulk AR(1) (4 x 16,384) | 9.29 ms | 16.00 ms | 1.72x | faster |
| HDI nearest (1,000,000) | 24.43 ms | 26.31 ms | 1.08x | faster |

To reproduce:

```bash
pixi run bench
```
