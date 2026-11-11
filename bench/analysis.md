# mojoarviz — ANALYZE phase: where the time actually goes

Measured 2026-11-11 on the machine below. Every number here comes from a run in this
session. Nothing in `src/` or `tests/` was changed.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 2 sockets × 18 cores, 90 MiB L3,
Linux x86_64, Python 3.13.14; mojoarviz 0.1.0, ArviZ 1.2.0, NumPy 2.5.1,
arviz-stats 0.3.0. The host is shared: run-to-run spread on the 20–100 ms rows is up to
±50 % (see "Noise and method"), and achieved numpy copy bandwidth was measured at
**16–26 GB/s**, far below this CPU's ~60 GB/s peak.

Tools used (scratch, outside the repo): `pixi run bench` for the headline rows, then
`/tmp/mav-analysis/analysis.py` (upstream + numpy baselines, phase splits, allocation
counts), `attrib.py` / `kern.py` / `ranksplit.py` / `hdi.py` (glue-vs-kernel attribution),
and cProfile via `MODE=pprof`.

## Headline: `pixi run bench`

Best of five warmed runs, three consecutive session runs (first run of the session, then
two more). The ratio is **ArviZ / Mojo**, i.e. above 1 means mojoarviz is faster.

| case | mojoarviz | ArviZ | ratio | result |
|---|---:|---:|---:|---|
| R-hat identity (4 x 1,000,000) | 22.38 / 34.27 / 22.02 ms | 39.62 / 52.77 / 40.63 ms | 1.77x / 1.54x / 1.84x | faster |
| R-hat rank (4 x 100,000) | 113.33 / 120.80 / 116.43 ms | 172.01 / 171.83 / 176.03 ms | 1.52x / 1.42x / 1.51x | faster |
| ESS identity AR(1) (4 x 16,384) | 2.81 / 2.84 / 2.72 ms | 4.86 / 5.09 / 4.90 ms | 1.73x / 1.79x / 1.80x | faster |
| ESS bulk AR(1) (4 x 16,384) | 8.86 / 10.27 / 10.33 ms | 15.70 / 16.48 / 16.28 ms | 1.77x / 1.60x / 1.58x | faster |
| HDI nearest (1,000,000) | 24.36 / 20.72 / 19.89 ms | 22.27 / 22.55 / 22.44 ms | 0.91x / 1.09x / 1.13x | slower / faster / faster |

The shipped benchmark does **not** show the two real regressions found below: both sit at
sizes outside its five cases. Note also that HDI nearest is effectively a tie and flips
result between runs (0.91x–1.13x); the README's stated 1.08x is inside that band.

## Ranked hotspot table

`flops/byte` is arithmetic intensity from the operation catalogue in
`bench/analysis.py` (bytes touched = input bytes read + buffers written once; nothing is
counted twice). "Sits in" says where the measured time is: **Python/ctypes glue**,
**kernel**, or the split between them, taken from the attribution runs.
A row marked **SLOWER** is slower than upstream ArviZ at that size.

Rows are ranked by where wall time is spent, so the same public call can appear twice
(e.g. the rank quicksort kernel and the upstream gap it causes). Byte counts and flops are
for one *execution of that row's operation*, not for the whole public call; the call-level
totals are in the "Kernel vs glue" table below.

| # | function | input size | time | bytes touched | flops | flops/byte | bound | sits in | verdict |
|---|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `rhat` `method="rank"` — quicksort, one pass | 4 x 100,000 (split → 8 x 50,000) | 49.7–50.3 ms per pass; pair 95.1–104.1 ms; 134.8 ms whole call | 4.8 MB | 2.0 M | 0.42 | compute-bound | kernel (sort + per-tie serial loop) | simd, then parallel |
| 2 | same kernel vs numpy and vs upstream | same | 49.7 ms vs `np.argsort` 50.9 ms vs ArviZ `_compute_ranks` 37.2 ms | 4.8 MB | 2.0 M | 0.42 | compute-bound | kernel | **SLOWER** than upstream's rank pass (kernel itself); whole call still 1.42–1.52x faster than ArviZ |
| 3 | `ess` `method="identity"` — numpy FFT above the 65,536 padded-value cutoff | 8 x 100,000 (padded 8 x 262,144) | **98.0 ms vs ArviZ 64.8 ms — SLOWER** (4-run range 79.6–98.0 vs 50.9–64.8) | 12.8 MB | 72.0 M | 5.62 | memory-bound | Python/NumPy — the FFT is not in the kernel | none for the kernel; Python-side FFT sizing (row 4) |
| 4 | same call — FFT length choice | same | numpy pipeline n=262,144: 90.6 ms; n=200,000: 55.0 ms | 12.8 MB | 72.0 M | 5.62 | memory-bound | Python/NumPy | none — 35.6 ms is the power-of-two padding |
| 5 | `ess` `method="bulk"` — rank pass | 4 x 16,384 (split → 8 x 16,384) | 7.3 ms of a 9.0–10.1 ms call | 0.5 MB | 1.0 M | 2.11 | compute-bound | kernel (`rank_normalize`) | unroll (sort); the native FFT beside it is row 10 |
| 6 | `hdi` on batched rows — Python loop re-entering ctypes | 1,000 x 1,000 (`axis=-1`) | **38.1 ms vs ArviZ 26.6 ms — SLOWER** | 16.0 MB | 3.0 M | 0.19 | memory-bound | Python/ctypes glue (96 %) | none for the GPU; unroll the Python loop |
| 7 | `rhat` `method="identity"` — Mojo reduction | 4 x 1,000,000 | 21.4 ms call; kernel 16.7–17.8 ms | 32.0 MB | 28.0 M | 0.87 | memory-bound | kernel | simd |
| 8 | same, vs numpy-only baseline | same | kernel 17.8 ms vs numpy `var(axis=1)` 15.7 ms, numpy 2-pass 28.6 ms | 32.0 MB | 28.0 M | 0.87 | memory-bound | kernel | `var` is not the ceiling; the kernel's own pass structure is (row 9) |
| 9 | same — kernel pass structure | same | 17.8 ms for 32 MB ≈ 1.8 GB/s effective (numpy `mean` alone: 2.33 ms) | 48.0 MB (3 reads of the buffer) | 28.0 M | 0.58 | memory-bound | kernel | simd: scalar accumulator chains, 3 passes where 1 suffices |
| 10 | `ess` `method="identity"` — native FFT + finish | 4 x 16,384 | 2.2–4.6 ms kernel in a 2.7–5.0 ms call | 0.5 MB | 4.9 M | 9.38 | memory-bound | kernel | gpu (only row above 2 flops/byte) |
| 11 | same call — `parallelize` effect | 4 x 16,384 vs 1 x 16,384 unit | kernel 4.62 ms; serial unit 1.91 ms → 4x serial 7.65 ms → **1.66x** | 0.5 MB | 4.9 M | 9.38 | memory-bound | kernel | parallel already active; more threads did not scale past 1.66x here |
| 12 | `rhat` `method="identity"` — validity glue | 4 x 1,000,000 | `_valid` 1.8 ms + `isfinite().all()` 2.0 ms = 3.8 ms of 21.4 ms | 64.0 MB (two NaN scans) | 0 | 0 | memory-bound | Python/NumPy glue | none — drop one of the two scans |
| 13 | `hdi` nearest, single flat array | 1,000,000 | 18.3–24.8 ms; numpy sort-only 18.0–18.5 ms *(vs ArviZ 19.8–24.4 ms)* | 16.0 MB | 3.0 M | 0.19 | memory-bound | Python (numpy sort 97 %) | none |
| 14 | `rhat` `method="split"` — reduction only | 4 x 100,000 (split → 8 x 50,000) | 2.21 ms *(vs ArviZ 4.68 ms)* | 3.2 MB | 2.8 M | 0.87 | memory-bound | kernel 1.59 ms + glue 0.62 ms | none (already faster than upstream) |

**SLOWER rows:** 2 (kernel vs upstream's rank pass), 3 (public `ess` identity at 8 x 1e5),
6 (batched HDI). Row 13 is marked as losing to numpy while tying upstream. Sizes where the
port beats upstream are not marked.

### Kernel vs glue, per public call (measured)

| call | total | in Python/NumPy/ctypes glue | in kernel(s) | glue share |
|---|---:|---:|---:|---:|
| `rhat` rank 4 x 100,000 | 110.6–134.8 ms (bench: 113.3–120.8) | 9.2 ms measured: median 5.36, folded buffer 3.20, `_rank_prepare` 0.65; ~10 pre-call ctypes entries add ≪1 ms | 95.1–104.1 ms pair, 3.19 ms reduce | ~8 % |
| `rhat` identity 4 x 1,000,000 | 21.4 ms | 3.8 ms validity + ~0.01 ms ctypes | 16.7–17.8 ms | 18 % |
| `ess` identity 8 x 100,000 | 98.0 ms | 96 ms (centering 0.95, numpy FFT 90.6, checks 1.0) | finish kernel, sub-ms | ~98 % |
| `ess` bulk 4 x 16,384 | 9.0–10.1 ms | ~0.1 ms | `rank_normalize` 7.3 + `mav_ess` 2.59 | ~1 % |
| `ess` identity 4 x 16,384 | 2.7–5.0 ms | 0.08 ms (scratch alloc 0.9 µs, ptp+isfinite 76 µs) | 2.2–4.6 ms | 2 % |
| `hdi` 1,000 x 1,000 | 38.1 ms | 36.6 ms (1,000 × `_hdi_one` at 55.8 µs/row) | ~1.5 ms (1,000 × ≤10 µs) | 96 % |
| `hdi` 1,000,000 flat | 18.3–24.4 ms | 18.0–18.5 ms (`np.sort` alone) | ~0.6 ms | ~97 % |

The `ess` identity 4 x 16,384 row varies 2.7–5.0 ms between runs while its kernel varies
2.2–4.6 ms; the glue is stable at 0.08 ms, so the spread is the native FFT kernel on a
loaded host, not the Python layer.

One ctypes round trip costs **~0.5–1.3 µs**: the empty `mav_hdi_nearest` call measured
9.82 µs at repeat=500 in one run and 1.25 µs in another, so the host's cache state swings
it by ~8x. Per call that is negligible everywhere, including batched HDI (1,000 × ~1 µs
≈ 1 ms of the 38.1 ms); the Python work per row, not the ctypes entry, is what costs.

## Python-level loops that re-enter ctypes per element

- `python/mojoarviz/diagnostics.py:347` — `_hdi_array` does
  `for index, values in enumerate(batch): result[index] = _hdi_one(values, ...)`, and
  `_hdi_one` → `_lib.hdi_nearest` → `python/mojoarviz/_lib.py:239` calls
  `lib().mav_hdi_nearest(...)` **once per batch row** (also `np.ascontiguousarray(row)`
  and `np.sort` per row). Measured: 55.8 µs per row end-to-end, of which 10.9 µs is the
  sort and ≤10 µs the kernel; total 38.1 ms for 1,000 rows vs 11.5 ms for a vectorised
  numpy equivalent. This is the clearest "Python loop around ctypes" in the package.
- `python/mojoarviz/diagnostics.py:131` — `_map_samples` loops over samples **in Python**
  but each element re-enters ctypes once (through `_ess_one`/`_rhat_one`), not per value.
  Only a problem when `batch_shape` is large; the batched-HDI shape is the one measured
  above.
- Everything else calls the shared library a constant number of times per public call:
  `core_rhat` 1 (plus `isfinite`), `rank_normalize` 1, `rank_normalize_pair` 1,
  `core_ess` 2 below the cutoff (validity already done in Python; the native call carries
  the reduced core into the same chain) or 2 above it. No per-draw or per-value ctypes
  traffic was found.

## Per-call allocations and copies (tracemalloc, one call)

| case | allocations | peak traced bytes | copies made |
|---|---:|---:|---|
| `rhat` rank 4 x 100,000 | 12 | 39.6 MB | `_split_chains` (3.2 MB), `np.median` working copy, `np.abs(...)` folded buffer, 8 rank buffers |
| `rhat` identity 4 x 1,000,000 | 6 | 4.0 MB | none large; `isfinite` allocates a **9.2 MB** boolean internally (untraced) |
| `ess` identity 4 x 16,384 | 7 | 2.2 MB | scratch `2 × chains × fft_n` = 2.1 MB allocated **per call** (`_lib.py:192`) |
| `ess` bulk 8 x 100,000 | 12 | 46.8 MB | split concat, `np.abs` folded, rank buffers, two full rfft/irfft spectra |
| `hdi` 1,000,000 | 5 | 8.0 MB | full `np.sort` copy passed zero-copy to the kernel (good) |
| `hdi` 1,000 x 1,000 | 5 | 28 KB | traced total hides the 1,000 per-row temporaries: `np.sort`, `reshape`, `np.ascontiguousarray` per row |

The `mav_ess` path re-allocates its FFT scratch on every call (`np.empty(2*chains*fft_n)`
at `_lib.py:192`); at 4 x 16,384 that is 0.9 µs, so it is not a hotspot today, but it
scales with chains and is the first thing to hit if ESS parallelism grows.

## Next phase per hotspot

| hotspot | next phase | why, with the number behind it |
|---|---|---|
| Mojo quicksort in `rank_normalize` (rows 1, 2, 5) | **simd**, then **parallel** | 0.42 flops/byte, 49.7–50.3 ms per 400k-element sort — level with scalar `np.argsort` (50.9 ms) and 1.34x slower than ArviZ's rank pass (37.2 ms). Not GPU: integer comparisons plus a serial tie loop (`dest[Int(indices[i])] = z`) that does not vectorise onto a GPU without a full sort rewrite, and 0.42 flops/byte is far below the 2 flops/byte gate. |
| `mav_rhat` reduction (rows 7–9) | **simd** | 0.87 flops/byte, 17.8 ms for 32 MB ≈ 1.8 GB/s effective against 16–26 GB/s measured numpy bandwidth. The kernel is scalar with three reads of the buffer and single-accumulator dependency chains; host-width SIMD is the direct fix. Not GPU at 0.87 flops/byte. |
| native FFT in `mav_ess` (rows 10, 11) | **gpu** (call 1) | 9.38 flops/byte — the only hotspot above the 2 flops/byte threshold — and it is a dense FFT, the one shape a GPU suits here. Measured caveat: the existing `parallelize` path already gives 1.66x (4.62 vs 7.65 ms) at 4 x 16,384, so a GPU verdict applies to many-chain jobs; the launch plus copy arithmetic for that shape was not measured and should be before a GPU phase starts. |
| numpy-FFT ESS path (rows 3, 4) | **none** | 5.62 flops/byte, but 90.6 of 98.0 ms is `np.fft` at n=262,144; n=200,000 costs 55.0 ms. That is a Python-side size choice, not a kernel. A GPU verdict would need copy-in + copy-out (12.8 MB + 12.8 MB) against that FFT work; not measured, so **no GPU verdict**. |
| batched HDI Python loop (row 6) | **unroll** (Python loop); **parallel** only if rows stay large | 0.19 flops/byte, nowhere near GPU. 36.6 of 38.1 ms is Python; kernel ≤10 µs/row. A vectorised numpy sort+argmin measured 11.5 ms, which removes the gap without touching Mojo. |
| flat HDI (row 13) | **none** | 0.19 flops/byte; 18.0–18.5 of 18.3–24.4 ms is numpy's own sort. |
| `rhat` split reduction (row 14) | **none** | Already 2.21 ms vs upstream 4.68 ms; 0.87 flops/byte, kernel 1.59 of 2.21 ms. |
| validity double scan (row 12) | **none** | Python-side; two NaN scans of one buffer. |

## Behaviour found while measuring (not an algorithm change, but material)

1. **`mav_ess_from_acov` stride.** `_lib.py:199-206` passes `fft_n` (262,144 in the
   `8 x 100,000` case) as the acov row stride, and the kernel uses it as
   `acov[chain * acov_stride + t]` (`src/diagnostics.mojo:441-453`). The acov buffer does
   have `fft_n` columns, so the public path stays in bounds. Passing a narrower, correctly
   contiguous buffer (`draws` columns) with the same stride reads past the end of the
   allocation and produced a **hard segfault** in this session (exit 139), and before it
   crashed once the values it read changed the result (9,246.9 vs 20,315.4 from the same
   data at stride 100,000).
2. **A segfault I could not reproduce deterministically.** The first crash was a call with
   a zero-filled 8 x 262,144 buffer and the same stride; the identical call later
   returned a value. Rebuilding the library from the current `src/` produced a .so that is
   byte-identical (sha256 `0209d5b3…`) to the shipped `dist/libmojo-arviz.so`, so this is
   not a stale binary. Because it is not reproducible, I did not report a timing for that
   entry point and I am not claiming a defect. It is the one thing in this phase I would
   re-run under a debugger before the next phase touches `ess`.
3. **Port agreement with upstream was checked, not assumed.** For every constructed case
   (`8 x 1e5`, `4 x 1e5`, `2 x 7e4` rho=0.5, `4 x 16384`), `maz.ess(x, method="identity")`
   matched `az.ess(...)` to the printed digits, e.g. 42789.29727198 vs 42789.2973.
   Independent memory pressure earlier in my harness changed kernel *results* through
   unrelated stride experiments; that is my harness, not the port, so none of those
   numbers are in the table.
4. **Method notes.** `arviz-stats 0.3.0` no longer exposes `_rhat`/`_ess`, so the
   "numpy-only" column uses the same formulas written in numpy (`mean`, `var(ddof=1)`,
   a numpy FFT-based autocovariance), plus arviz-stats' own `_compute_ranks` and
   `autocov`. Allocation counts come from `tracemalloc` snapshots taken before and after
   one warm call. Times are the best of 7–11 repeats; the HDI 1e6 row is marked because
   it loses to numpy and ties upstream across runs (0.91x–1.13x reported for the same
   case).
