"""ctypes bindings for the compiled Mojo diagnostic kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("MOJOARVIZ_LIB", ROOT / "dist" / "libmojo-arviz.so"))
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mav_rank_normalize": ([I, I, I, I, I], None),
    "mav_rank_normalize_pair": ([I, I, I, I, I, I, I, I, I], None),
    "mav_rhat": ([I, I, I], F),
    "mav_ess": ([I, I, I, I, I, I, I, I], F),
    "mav_ess_from_acov": ([I, I, I, I, I, I, I], F),
    "mav_hdi_nearest": ([I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> Path:
    if os.environ.get("MOJOARVIZ_LIB") and LIB.exists() and not force:
        return LIB
    sources = list((ROOT / "src").glob("*.mojo"))
    if not sources:
        if LIB.exists():
            return LIB
        raise BuildError(f"no Mojo sources found under {ROOT / 'src'}")
    if not force and LIB.exists() and LIB.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo is not on PATH; run `pixi run build` first")
    LIB.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "shared-lib",
            str(ROOT / "src" / "diagnostics.mojo"),
            "-o",
            str(LIB),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not LIB.exists():
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if array.dtype != np.float64 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)


def _matrix(values: np.ndarray, name: str = "values") -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional chain-by-draw array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one chain and one draw")
    return array


def rank_normalize(values: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float64)
    flat = source.reshape(-1)
    if flat.size == 0:
        raise ValueError("rank normalization requires at least one value")
    sorted_values = np.empty_like(flat)
    indices = np.empty_like(flat)
    result = np.empty_like(flat)
    lib().mav_rank_normalize(
        addr(flat), addr(sorted_values), addr(indices), addr(result), flat.size
    )
    return result.reshape(source.shape)


def rank_normalize_pair(
    first_values: np.ndarray, second_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first_source = np.ascontiguousarray(first_values, dtype=np.float64)
    second_source = np.ascontiguousarray(second_values, dtype=np.float64)
    if first_source.shape != second_source.shape:
        raise ValueError("rank normalization inputs must have the same shape")
    first_flat = first_source.reshape(-1)
    second_flat = second_source.reshape(-1)
    if first_flat.size == 0:
        raise ValueError("rank normalization requires at least one value")
    first_sorted = np.empty_like(first_flat)
    first_indices = np.empty_like(first_flat)
    first_result = np.empty_like(first_flat)
    second_sorted = np.empty_like(second_flat)
    second_indices = np.empty_like(second_flat)
    second_result = np.empty_like(second_flat)
    lib().mav_rank_normalize_pair(
        addr(first_flat),
        addr(first_sorted),
        addr(first_indices),
        addr(first_result),
        addr(second_flat),
        addr(second_sorted),
        addr(second_indices),
        addr(second_result),
        first_flat.size,
    )
    return (
        first_result.reshape(first_source.shape),
        second_result.reshape(second_source.shape),
    )


def core_rhat(values: np.ndarray) -> float:
    array = _matrix(values)
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("R-hat requires at least two chains and two draws")
    return float(lib().mav_rhat(addr(array), array.shape[0], array.shape[1]))


def core_ess(values: np.ndarray, relative: bool = False) -> float:
    array = _matrix(values)
    chains, draws = array.shape
    if draws < 2:
        raise ValueError("ESS requires at least two draws")
    if np.ptp(array) < 1.0e-15:
        return float(array.size)
    fft_n = 1
    while fft_n < 2 * draws:
        fft_n <<= 1
    rho = np.empty(draws, dtype=np.float64)
    if array.size > 65536:
        centered = array - array.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered, n=fft_n, axis=1)
        spectrum.real *= spectrum.real
        spectrum.imag *= spectrum.imag
        spectrum.real += spectrum.imag
        spectrum.imag.fill(0.0)
        acov = np.fft.irfft(spectrum, n=fft_n, axis=1)
        return float(
            lib().mav_ess_from_acov(
                addr(array),
                addr(acov),
                addr(rho),
                chains,
                draws,
                fft_n,
                int(relative),
            )
        )
    scratch = np.empty(2 * chains * fft_n, dtype=np.float64)
    real = scratch[: chains * fft_n]
    imag = scratch[chains * fft_n :]
    return float(
        lib().mav_ess(
            addr(array),
            addr(real),
            addr(imag),
            addr(rho),
            chains,
            draws,
            fft_n,
            int(relative),
        )
    )


def hdi_nearest(values: np.ndarray, interval_idx: int) -> np.ndarray:
    work = np.sort(np.asarray(values, dtype=np.float64), axis=None)
    if work.size == 0:
        raise ValueError("HDI requires at least one value")
    if not 0 <= interval_idx < work.size:
        raise ValueError("HDI interval index is outside the input buffer")
    result = np.empty(2, dtype=np.float64)
    lib().mav_hdi_nearest(addr(work), addr(result), work.size, interval_idx)
    return result
