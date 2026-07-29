"""Honest mojoarviz benchmarks against ArviZ on identical arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import arviz as az  # noqa: E402
import mojoarviz as maz  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def ar1(chains, draws, rho, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(chains, draws))
    values = np.empty_like(noise)
    values[:, 0] = noise[:, 0]
    for index in range(1, draws):
        values[:, index] = rho * values[:, index - 1] + noise[:, index]
    return values


def machine():
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(42)
    cases = [
        (
            "R-hat identity (4 x 1,000,000)",
            lambda x=np.ascontiguousarray(rng.normal(size=(4, 1_000_000))): (
                lambda: maz.rhat(x, method="identity"),
                lambda: az.rhat(x, method="identity"),
            ),
        ),
        (
            "R-hat rank (4 x 100,000)",
            lambda x=np.ascontiguousarray(rng.normal(size=(4, 100_000))): (
                lambda: maz.rhat(x),
                lambda: az.rhat(x),
            ),
        ),
        (
            "ESS identity AR(1) (4 x 16,384)",
            lambda x=ar1(4, 16_384, 0.9, 1): (
                lambda: maz.ess(x, method="identity"),
                lambda: az.ess(x, method="identity"),
            ),
        ),
        (
            "ESS bulk AR(1) (4 x 16,384)",
            lambda x=ar1(4, 16_384, 0.9, 2): (
                lambda: maz.ess(x),
                lambda: az.ess(x),
            ),
        ),
        (
            "HDI nearest (1,000,000)",
            lambda x=np.ascontiguousarray(rng.standard_t(5, size=1_000_000)): (
                lambda: maz.hdi(x, prob=0.94),
                lambda: az.hdi(x, prob=0.94),
            ),
        ),
    ]

    print(f"Machine: {machine()}")
    print(f"Versions: mojoarviz {maz.__version__}; ArviZ {az.__version__}; NumPy {np.__version__}")
    print()
    print("| case | mojoarviz | ArviZ | ArviZ / Mojo | result |")
    print("|---|---:|---:|---:|---|")
    for name, setup in cases:
        ours, upstream = setup()
        ours_value = ours()
        upstream_value = upstream()
        np.testing.assert_allclose(
            np.asarray(ours_value),
            np.asarray(upstream_value),
            rtol=1e-7,
            atol=1e-7,
            equal_nan=True,
        )
        ours_time = timeit(ours)
        upstream_time = timeit(upstream)
        ratio = upstream_time / ours_time
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | {upstream_time * 1e3:.2f} ms "
            f"| {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
