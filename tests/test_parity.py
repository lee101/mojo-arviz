"""Numerical and behavioral parity against the installed ArviZ release."""

from __future__ import annotations

import ctypes
import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

az = pytest.importorskip("arviz")
xr = pytest.importorskip("xarray")

import mojoarviz as maz
from mojoarviz._lib import addr, core_ess, core_rhat, hdi_nearest


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(20260729)
    values = rng.normal(size=(4, 501, 3))
    values[:, :, 1] += np.linspace(-0.15, 0.15, 4)[:, None]
    values[:, :, 2] = np.round(values[:, :, 2], 1)
    return values


def assert_close(ours, upstream, *, atol=1e-7):
    np.testing.assert_allclose(np.asarray(ours), np.asarray(upstream), rtol=1e-9, atol=atol)


def test_public_signatures_match():
    assert inspect.signature(maz.ess) == inspect.signature(az.ess)
    assert inspect.signature(maz.rhat) == inspect.signature(az.rhat)
    assert inspect.signature(maz.hdi) == inspect.signature(az.hdi)


def test_ffi_buffer_contracts_reject_unsafe_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        addr(np.empty(0, dtype=np.float64))
    with pytest.raises(TypeError, match="C-contiguous float64"):
        addr(np.ones(4, dtype=np.float32))
    with pytest.raises(TypeError, match="C-contiguous float64"):
        addr(np.ones(8, dtype=np.float64)[::2])
    with pytest.raises(ValueError, match="two-dimensional"):
        core_ess(np.ones(8))
    with pytest.raises(ValueError, match="two chains"):
        core_rhat(np.ones((1, 8)))
    with pytest.raises(ValueError, match="outside"):
        hdi_nearest(np.ones(8), 8)


@pytest.mark.parametrize(
    ("method", "prob"),
    [
        ("bulk", None),
        ("tail", 0.1),
        ("mean", None),
        ("sd", None),
        ("median", None),
        ("mad", None),
        ("z_scale", None),
        ("folded", None),
        ("identity", None),
        ("quantile", 0.3),
        ("local", (0.2, 0.8)),
    ],
)
def test_ess_all_methods(draws, method, prob):
    kwargs = {} if prob is None else {"prob": prob}
    assert_close(
        maz.ess(draws, method=method, **kwargs),
        az.ess(draws, method=method, **kwargs),
    )


@pytest.mark.parametrize("method", ["rank", "identity", "split", "z_scale", "folded"])
def test_rhat_all_methods(draws, method):
    assert_close(maz.rhat(draws, method=method), az.rhat(draws, method=method), atol=2e-10)


def test_ess_relative(draws):
    for method in ("bulk", "mean", "identity", "quantile"):
        kwargs = {"prob": 0.65} if method == "quantile" else {}
        assert_close(
            maz.ess(draws, method=method, relative=True, **kwargs),
            az.ess(draws, method=method, relative=True, **kwargs),
            atol=2e-10,
        )


def test_single_chain_with_chain_axis_none():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(2, 701))
    assert_close(
        maz.ess(values, chain_axis=None, draw_axis=1),
        az.ess(values, chain_axis=None, draw_axis=1),
    )


def test_nondefault_and_negative_axes():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(2, 401, 4))
    assert_close(
        maz.ess(values, chain_axis=-1, draw_axis=1),
        az.ess(values, chain_axis=-1, draw_axis=1),
    )
    assert_close(
        maz.rhat(values, chain_axis=-1, draw_axis=1),
        az.rhat(values, chain_axis=-1, draw_axis=1),
        atol=2e-10,
    )


def test_autocorrelated_ess():
    rng = np.random.default_rng(6)
    noise = rng.normal(size=(4, 2000))
    values = np.empty_like(noise)
    values[:, 0] = noise[:, 0]
    for draw in range(1, values.shape[1]):
        values[:, draw] = 0.92 * values[:, draw - 1] + noise[:, draw]
    assert_close(maz.ess(values), az.ess(values), atol=2e-6)
    assert_close(maz.ess(values, method="identity"), az.ess(values, method="identity"), atol=1e-8)


@pytest.mark.parametrize(("chains", "draw_count"), [(3, 8191), (4, 8193)])
def test_ess_simd_tail_and_native_parallel_threshold(chains, draw_count):
    rng = np.random.default_rng(61)
    values = rng.normal(size=(chains, draw_count))
    assert_close(
        maz.ess(values, method="identity"),
        az.ess(values, method="identity"),
        atol=1e-8,
    )


def test_ess_batched_fft_threshold():
    rng = np.random.default_rng(62)
    noise = rng.normal(size=(4, 16385))
    values = np.empty_like(noise)
    values[:, 0] = noise[:, 0]
    for draw in range(1, values.shape[1]):
        values[:, draw] = 0.85 * values[:, draw - 1] + noise[:, draw]
    assert_close(
        maz.ess(values, method="identity"),
        az.ess(values, method="identity"),
        atol=1e-8,
    )


def test_rank_rhat_parallel_threshold():
    rng = np.random.default_rng(63)
    values = rng.normal(size=(4, 16385))
    assert_close(maz.rhat(values), az.rhat(values), atol=2e-10)


def test_tied_values():
    rng = np.random.default_rng(7)
    values = rng.integers(-3, 4, size=(4, 500))
    assert_close(maz.ess(values), az.ess(values))
    assert_close(maz.rhat(values), az.rhat(values), atol=2e-10)


def test_constant_values():
    values = np.ones((4, 100))
    assert maz.ess(values) == az.ess(values) == 400
    assert np.isnan(maz.rhat(values))
    with np.errstate(invalid="ignore"):
        assert np.isnan(az.rhat(values))


def test_too_few_draws_and_chains_return_nan():
    assert np.isnan(maz.ess(np.ones((1, 3))))
    assert np.isnan(maz.rhat(np.ones((1, 20))))
    assert np.isnan(maz.rhat(np.ones((2, 3))))


def test_nonfinite_diagnostics_return_nan():
    values = np.ones((4, 20))
    values[1, 4] = np.nan
    assert np.isnan(maz.ess(values))
    assert np.isnan(maz.rhat(values))


@pytest.mark.parametrize("method", ["tail", "quantile", "local"])
def test_required_ess_probability(method):
    with pytest.raises(TypeError):
        maz.ess(np.arange(80.0).reshape(4, 20), method=method)


def test_local_probability_validation():
    with pytest.raises(ValueError, match="upper and lower"):
        maz.ess(np.arange(80.0).reshape(4, 20), method="local", prob=(0.5,))


def test_invalid_diagnostic_methods():
    with pytest.raises(ValueError):
        maz.ess(np.ones((4, 20)), method="unknown")
    with pytest.raises(ValueError):
        maz.rhat(np.ones((4, 20)), method="unknown")


@pytest.mark.parametrize("prob", [0.01, 0.5, 0.8, 0.94])
def test_hdi_vector(prob):
    rng = np.random.default_rng(10)
    values = rng.standard_t(4, size=1001)
    assert_close(maz.hdi(values, prob=prob), az.hdi(values, prob=prob), atol=0)


@pytest.mark.parametrize("axis", [-1, 0, (0, 1), None])
def test_hdi_axes(axis):
    rng = np.random.default_rng(11)
    values = rng.normal(size=(4, 101, 3))
    assert_close(
        maz.hdi(values, prob=0.83, axis=axis),
        az.hdi(values, prob=0.83, axis=axis),
        atol=0,
    )


def test_hdi_default_probability():
    values = np.arange(100.0)
    assert_close(maz.hdi(values), az.hdi(values), atol=0)


def test_hdi_single_value():
    values = np.array([3.5])
    assert_close(maz.hdi(values), az.hdi(values), atol=0)


def test_hdi_skipna():
    values = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert_close(maz.hdi(values, prob=0.6, skipna=True), az.hdi(values, prob=0.6, skipna=True))
    assert_close(
        maz.hdi(values, prob=0.6, skipna=False),
        az.hdi(values, prob=0.6, skipna=False),
        atol=0,
    )


def test_hdi_circular():
    rng = np.random.default_rng(12)
    values = np.angle(np.exp(1j * rng.normal(np.pi - 0.1, 0.25, size=2000)))
    assert_close(
        maz.hdi(values, prob=0.8, circular=True),
        az.hdi(values, prob=0.8, circular=True),
        atol=1e-12,
    )


def test_hdi_validation_and_unsupported_modes():
    with pytest.raises(ValueError):
        maz.hdi(np.arange(10), prob=0)
    with pytest.raises(ValueError):
        maz.hdi(np.arange(10), prob=1)
    with pytest.raises(NotImplementedError):
        maz.hdi(np.arange(10), method="multimodal")


def test_dataarray_parity(draws):
    array = xr.DataArray(
        draws,
        dims=("chain", "draw", "school"),
        coords={"school": ["a", "b", "c"]},
        name="theta",
    )
    ours_ess = maz.ess(array)
    theirs_ess = az.ess(array)
    assert ours_ess.dims == theirs_ess.dims
    assert ours_ess.coords.equals(theirs_ess.coords)
    assert_close(ours_ess, theirs_ess)

    ours_rhat = maz.rhat(array)
    theirs_rhat = az.rhat(array)
    assert ours_rhat.dims == theirs_rhat.dims
    assert_close(ours_rhat, theirs_rhat, atol=2e-10)

    ours_hdi = maz.hdi(array, prob=0.8)
    theirs_hdi = az.hdi(array, prob=0.8)
    assert ours_hdi.dims == theirs_hdi.dims
    assert list(ours_hdi.ci_bound.values) == ["lower", "upper"]
    assert_close(ours_hdi, theirs_hdi, atol=0)


def test_dataarray_custom_sample_dimension_names():
    rng = np.random.default_rng(64)
    array = xr.DataArray(
        rng.normal(size=(4, 301, 2)),
        dims=("walker", "sample", "parameter"),
    )
    sample_dims = ("walker", "sample")
    assert_close(
        maz.ess(array, sample_dims=sample_dims),
        az.ess(array, sample_dims=sample_dims),
    )
    assert_close(
        maz.rhat(array, sample_dims=sample_dims),
        az.rhat(array, sample_dims=sample_dims),
        atol=2e-10,
    )


def test_dataset_selection_and_coords(draws):
    dataset = xr.Dataset(
        {
            "theta": (("chain", "draw", "school"), draws),
            "theta_aux": (("chain", "draw", "school"), draws + 0.5),
            "mu": (("chain", "draw"), draws[:, :, 0]),
        },
        coords={"school": ["a", "b", "c"]},
    )
    ours = maz.ess(
        dataset,
        var_names=["theta"],
        filter_vars="like",
        coords={"school": ["a", "c"]},
    )
    theirs = az.ess(
        dataset,
        var_names=["theta"],
        filter_vars="like",
        coords={"school": ["a", "c"]},
    )
    assert set(ours.data_vars) == set(theirs.data_vars)
    for name in ours:
        assert_close(ours[name], theirs[name])


def test_dataset_hdi(draws):
    dataset = xr.Dataset(
        {
            "theta": (("chain", "draw", "school"), draws),
            "mu": (("chain", "draw"), draws[:, :, 0]),
        }
    )
    ours = maz.hdi(dataset, prob=0.75)
    theirs = az.hdi(dataset, prob=0.75)
    assert set(ours.data_vars) == set(theirs.data_vars)
    for name in ours:
        assert_close(ours[name], theirs[name], atol=0)


# ---------------------------------------------------------------------------
# Parity of the FFI-level kernels against the very same internals ArviZ uses.
# ---------------------------------------------------------------------------

from arviz_stats.base.core import _CoreBase  # noqa: E402
from arviz_stats.base.diagnostics import _DiagnosticsBase  # noqa: E402

from mojoarviz._lib import rank_normalize, rank_normalize_pair  # noqa: E402


@pytest.fixture(scope="module")
def upstream_base():
    return _CoreBase(), _DiagnosticsBase()


def test_rank_normalization_matches_upstream_z_scale(upstream_base):
    _, diagnostics = upstream_base
    rng = np.random.default_rng(101)
    cases = [
        rng.normal(size=(1, 50)),
        np.round(rng.normal(size=(4, 13)), 1),  # ties
        np.round(rng.normal(size=(3, 7, 2)), 1),  # batch + ties
        np.ones((2, 6)),  # all equal
    ]
    for values in cases:
        # upstream: _DiagnosticsBase._z_scale (rankdata + Blom back-transform)
        # atol covers Mojo's Acklam ppf vs scipy.stats.norm.ppf (observed max ~3e-9).
        assert_close(rank_normalize(values), diagnostics._z_scale(values), atol=2e-8)


def test_rank_normalization_ignores_nan_like_upstream(upstream_base):
    _, diagnostics = upstream_base
    rng = np.random.default_rng(102)
    values = rng.normal(size=(2, 60))
    values.ravel()[::7] = np.nan
    # atol covers Mojo's Acklam ppf vs scipy.stats.norm.ppf (observed max ~3e-9).
    assert_close(rank_normalize(values), diagnostics._z_scale(values), atol=2e-8)
    assert np.isnan(rank_normalize(values)).sum() == np.isnan(values).sum()


def test_rank_pair_matches_upstream_fold_and_z_scale(upstream_base):
    _, diagnostics = upstream_base
    split = np.round(np.random.default_rng(103).normal(size=(6, 21)), 1)
    folded = np.abs(split - np.median(split))
    bulk, tail = rank_normalize_pair(split, folded)
    assert_close(bulk, diagnostics._z_scale(split), atol=2e-8)
    assert_close(tail, diagnostics._z_scale(folded), atol=2e-8)


def test_core_kernels_match_upstream(upstream_base):
    core, diagnostics = upstream_base
    rng = np.random.default_rng(104)
    values = rng.normal(size=(4, 101))
    assert_close(core_ess(values), diagnostics._ess(values), atol=1e-9)
    assert_close(core_ess(values, True), diagnostics._ess(values, relative=True), atol=1e-9)
    assert_close(core_rhat(values), diagnostics._rhat(values), atol=1e-12)
    # constant input short-circuits to the sample count upstream.
    assert core_ess(np.ones((3, 9))) == diagnostics._ess(np.ones((3, 9))) == 27


def test_hdi_kernel_matches_upstream(upstream_base):
    core, _ = upstream_base
    rng = np.random.default_rng(105)
    values = rng.normal(size=(4, 101))
    flat = values.ravel()
    for prob in (0.3, 0.5, 0.89):
        interval_idx = int(np.floor(prob * flat.size))
        # upstream: _CoreBase._hdi_linear_nearest_common
        assert_close(
            hdi_nearest(values, interval_idx),
            core._hdi_linear_nearest_common(flat, prob),
            atol=0,
        )
    # ties: the first minimal window wins in both implementations.
    ties = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    for prob in (0.4, 0.6, 0.89):
        assert_close(
            hdi_nearest(ties, int(np.floor(prob * ties.size)), ravel=False),
            core._hdi_linear_nearest_common(ties, prob),
            atol=0,
        )


# ---------------------------------------------------------------------------
# Degenerate inputs: empty, one element, duplicates, non-finite, bad shapes.
#
# Every expectation below is taken from the installed ArviZ 1.2.0
# (arviz_stats 1.2.0) by calling it in the same test. Where upstream raises, the
# port must raise an exception of the same family; where upstream returns a
# value, the port must return that value within the stated tolerance.
# ---------------------------------------------------------------------------


def _call(function, *args, **kwargs):
    """Return ``("ok", value, None)`` or ``("error", None, exception_class)``."""
    try:
        return "ok", function(*args, **kwargs), None
    except Exception as err:  # noqa: BLE001 - parity is about the exception class
        return "error", None, type(err)


def _assert_same_error(ours, upstream):
    assert ours[0] == "error", ("port returned a value", ours)
    assert upstream[0] == "error", ("upstream returned a value", upstream)
    # numpy's AxisError subclasses ValueError, so this accepts upstream's
    # "axes don't match array" against the port's AxisError for 1-D input.
    assert issubclass(ours[2], (ValueError, KeyError)), ours[2]
    assert issubclass(upstream[2], (ValueError, KeyError)), upstream[2]


@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 4), (4, 0), (0, 0, 2), (3, 0, 2)])
@pytest.mark.parametrize("function", ["ess", "rhat"])
def test_empty_input_matches_upstream(shape, function):
    empty = np.empty(shape)
    ours = _call(getattr(maz, function), empty)
    upstream = _call(getattr(az, function), empty)
    if upstream[0] == "error":
        _assert_same_error(ours, upstream)
        return
    assert ours[0] == "ok", (shape, function, ours)
    assert np.shape(ours[1]) == np.shape(upstream[1])
    assert_close(ours[1], upstream[1], atol=0)


@pytest.mark.parametrize("method", ["bulk", "mean", "tail", "quantile", "local", "mad", "folded"])
def test_empty_input_matches_upstream_for_each_method(method):
    kwargs = {"tail": {"prob": 0.1}, "quantile": {"prob": 0.3}, "local": {"prob": (0.2, 0.8)}}
    empty = np.empty((0, 4))
    ours = _call(maz.ess, empty, method=method, **kwargs.get(method, {}))
    upstream = _call(az.ess, empty, method=method, **kwargs.get(method, {}))
    assert ours[0] == upstream[0] == "ok", (method, ours, upstream)
    # upstream: stats_utils._ufunc indexes the zero-size core result, giving NaN.
    assert np.isnan(ours[1]) and np.isnan(upstream[1])


def test_empty_hdi_matches_upstream():
    for values in (np.array([]), np.empty((2, 0))):
        ours = _call(maz.hdi, values)
        upstream = _call(az.hdi, values)
        _assert_same_error(ours, upstream)
        assert issubclass(upstream[2], ValueError)


def test_one_element_and_single_draw_match_upstream():
    rng = np.random.default_rng(106)
    for values in (np.array([1.0]), np.array([[1.0]]), rng.normal(size=(2, 1)), rng.normal(size=(1, 1))):
        for function in ("ess", "rhat"):
            ours = _call(getattr(maz, function), values)
            upstream = _call(getattr(az, function), values)
            assert ours[0] == upstream[0], (values.shape, function, ours, upstream)
            if upstream[0] == "ok":
                assert_close(ours[1], upstream[1], atol=0)


@pytest.mark.parametrize("draw_count", [1, 2, 3])
def test_too_few_draws_match_upstream(draw_count):
    values = np.random.default_rng(107).normal(size=(3, draw_count))
    for method in ("bulk", "identity", "median", "mad"):
        assert np.isnan(maz.ess(values, method=method))
        assert np.isnan(az.ess(values, method=method))
    for method in ("rank", "identity", "split", "folded", "z_scale"):
        assert np.isnan(maz.rhat(values, method=method))
        assert np.isnan(az.rhat(values, method=method))


def test_single_chain_matches_upstream():
    values = np.random.default_rng(108).normal(size=(1, 40))
    assert_close(maz.ess(values), az.ess(values), atol=1e-9)
    assert_close(maz.ess(values, method="identity"), az.ess(values, method="identity"), atol=1e-9)
    assert np.isnan(maz.rhat(values)) and np.isnan(az.rhat(values))


def test_duplicate_values_match_upstream():
    values = np.array([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]])
    for method in ("bulk", "identity", "mean", "sd", "median", "mad", "z_scale", "folded"):
        assert_close(maz.ess(values, method=method), az.ess(values, method=method), atol=1e-13)
    for method in ("rank", "identity", "split", "folded", "z_scale"):
        assert_close(maz.rhat(values, method=method), az.rhat(values, method=method), atol=1e-13)


def test_duplicate_values_hdi_match_upstream():
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    for prob in (0.4, 0.6, 0.89, 1.0):
        ours = _call(maz.hdi, values, prob=prob)
        upstream = _call(az.hdi, values, prob=prob)
        if upstream[0] == "error":
            _assert_same_error(ours, upstream)
            continue
        assert ours[0] == "ok", (prob, ours)
        assert_close(ours[1], upstream[1], atol=0)


def test_nan_inputs_match_upstream():
    values = np.arange(80.0).reshape(4, 20)
    values.ravel()[::9] = np.nan
    kwargs_for = {"tail": {"prob": 0.1}, "quantile": {"prob": 0.3}, "local": {"prob": (0.2, 0.8)}}
    for method in ("bulk", "identity", "mean", "sd", "median", "mad", "z_scale", "folded",
                   "tail", "quantile", "local"):
        kw = kwargs_for.get(method, {})
        assert np.isnan(maz.ess(values, method=method, **kw))
        assert np.isnan(az.ess(values, method=method, **kw))
    for method in ("rank", "identity", "split", "folded", "z_scale"):
        assert np.isnan(maz.rhat(values, method=method))
        assert np.isnan(az.rhat(values, method=method))


def test_all_nan_matches_upstream():
    values = np.full((4, 20), np.nan)
    assert np.isnan(maz.ess(values)) and np.isnan(az.ess(values))
    assert np.isnan(maz.rhat(values)) and np.isnan(az.rhat(values))


@pytest.mark.parametrize("position", [(0, 0), (1, 5), (3, 29)])
def test_infinite_draws_match_upstream(position):
    """Upstream's not_valid() rejects NaN but not inf, so rank-based methods stay finite."""
    values = np.random.default_rng(109).normal(size=(4, 30))
    values[position] = np.inf
    kwargs_for = {"tail": {"prob": 0.1}, "quantile": {"prob": 0.3}, "local": {"prob": (0.2, 0.8)}}
    for method in ("bulk", "mean", "mad", "folded", "z_scale", "median", "tail", "quantile", "local"):
        kw = kwargs_for.get(method, {})
        assert_close(
            maz.ess(values, method=method, **kw),
            az.ess(values, method=method, **kw),
            atol=1e-7,
        )
    for method in ("identity", "sd"):
        assert np.isnan(maz.ess(values, method=method))
        assert np.isnan(az.ess(values, method=method))
    for method in ("rank", "folded", "z_scale"):
        assert_close(maz.rhat(values, method=method), az.rhat(values, method=method), atol=2e-10)
    for method in ("identity", "split"):
        assert np.isnan(maz.rhat(values, method=method))
        assert np.isnan(az.rhat(values, method=method))


def test_infinite_left_tail_matches_upstream():
    values = np.random.default_rng(110).normal(size=(4, 30))
    values[2, 9] = -np.inf
    assert_close(maz.ess(values), az.ess(values), atol=1e-7)
    assert_close(maz.rhat(values), az.rhat(values), atol=2e-10)
    assert np.isnan(maz.ess(values, method="identity"))
    assert np.isnan(az.ess(values, method="identity"))


def test_mismatched_axes_match_upstream():
    values = np.random.default_rng(111).normal(size=(2, 10))
    for kwargs in (
        {"chain_axis": 1, "draw_axis": 1},
        {"chain_axis": 0, "draw_axis": 0},
        {"chain_axis": 5, "draw_axis": 1},
        {"chain_axis": 0, "draw_axis": -3},
    ):
        for function in ("ess", "rhat"):
            ours = _call(getattr(maz, function), values, **kwargs)
            upstream = _call(getattr(az, function), values, **kwargs)
            _assert_same_error(ours, upstream)


def test_1d_input_without_chain_axis_matches_upstream_error():
    values = np.random.default_rng(112).normal(size=20)
    ours = _call(maz.ess, values)
    upstream = _call(az.ess, values)
    # upstream raises ValueError("axes don't match array"); the port raises numpy's
    # AxisError, which is a ValueError subclass.
    _assert_same_error(ours, upstream)
    assert issubclass(ours[2], ValueError)
    assert_close(
        maz.ess(values, chain_axis=None, draw_axis=-1),
        az.ess(values, chain_axis=None, draw_axis=-1),
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([2.5]),
        np.array([2.0, 2.0]),
        np.array([]),
        np.empty((2, 0)),
        np.array([0.0, np.nan, 2.0, np.nan]),
        np.array([0.0, np.inf, 2.0, 3.0]),
        np.arange(12.0).reshape(3, 4),
    ],
)
@pytest.mark.parametrize("prob", [0.5, 0.89, 1.0])
def test_hdi_degenerate_inputs_match_upstream(values, prob):
    ours = _call(maz.hdi, values, prob=prob)
    upstream = _call(az.hdi, values, prob=prob)
    if upstream[0] == "error":
        _assert_same_error(ours, upstream)
        return
    assert ours[0] == "ok", (values.shape, prob, ours)
    assert np.shape(ours[1]) == np.shape(upstream[1])
    assert_close(ours[1], upstream[1], atol=0)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, bool])
def test_hdi_integer_and_bool_inputs_match_upstream(dtype):
    rng = np.random.default_rng(113)
    if dtype is bool:
        values = rng.integers(0, 2, size=101).astype(bool)
    else:
        values = rng.integers(-3, 4, size=101).astype(dtype)
    assert_close(maz.hdi(values, prob=0.6), az.hdi(values, prob=0.6), atol=0)


def test_hdi_nan_column_pattern_matches_upstream():
    values = np.random.default_rng(114).normal(size=(10, 3))
    values[:, 1] = np.nan
    for skipna in (False, True):
        assert_close(
            maz.hdi(values, prob=0.5, skipna=skipna),
            az.hdi(values, prob=0.5, skipna=skipna),
            atol=0,
        )


def test_hdi_circular_duplicates_and_one_element_match_upstream():
    rng = np.random.default_rng(115)
    for values in (
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([-np.pi + 0.01, np.pi - 0.01, 0.0]),
        np.array([1.25]),
        rng.uniform(-np.pi, np.pi, size=101),
    ):
        for prob in (0.5, 0.89):
            assert_close(
                maz.hdi(values, prob=prob, circular=True),
                az.hdi(values, prob=prob, circular=True),
                atol=1e-12,
            )


def test_hdi_axis_duplicates_and_kwargs_errors_match_upstream():
    values = np.random.default_rng(116).normal(size=(4, 11))
    ours = _call(maz.hdi, values, axis=(0, 0))
    upstream = _call(az.hdi, values, axis=(0, 0))
    if upstream[0] == "error":
        _assert_same_error(ours, upstream)
    else:
        assert ours[0] == "ok"
        assert_close(ours[1], upstream[1], atol=0)

    for kwargs in ({"foo": 1}, {"foo": 1, "bar": 2}):
        ours = _call(maz.hdi, values, **kwargs)
        upstream = _call(az.hdi, values, **kwargs)
        assert ours[0] == upstream[0] == "error", (kwargs, ours, upstream)
        assert ours[2] is upstream[2] is TypeError, (kwargs, ours[2], upstream[2])

    for prob in (0.0, 1.5, -0.1):
        ours = _call(maz.hdi, values, prob=prob)
        upstream = _call(az.hdi, values, prob=prob)
        assert ours[0] == upstream[0] == "error", (prob, ours, upstream)
        assert ours[2] is upstream[2] is ValueError, (prob, ours[2], upstream[2])


def test_hdi_default_probability_matches_upstream():
    values = np.random.default_rng(117).normal(size=200)
    assert_close(maz.hdi(values), az.hdi(values), atol=0)


def test_tail_and_quantile_probability_forms_match_upstream():
    values = np.random.default_rng(118).normal(size=(4, 60))
    for prob in (0.1, 0.5, 0.9, 1, [0.1, 0.9], (0.9, 0.1)):
        assert_close(
            maz.ess(values, method="tail", prob=prob),
            az.ess(values, method="tail", prob=prob),
            atol=1e-9,
        )
    for prob in (0.0, 0.5, 1.0):
        assert_close(
            maz.ess(values, method="quantile", prob=prob),
            az.ess(values, method="quantile", prob=prob),
            atol=1e-9,
        )


def test_missing_and_invalid_probability_errors_match_upstream():
    values = np.random.default_rng(119).normal(size=(4, 20))
    for method in ("tail", "quantile", "local"):
        ours = _call(maz.ess, values, method=method)
        upstream = _call(az.ess, values, method=method)
        assert ours[0] == upstream[0] == "error", (method, ours, upstream)
        assert ours[2] is upstream[2] is TypeError, (method, ours[2], upstream[2])
    for prob in ((0.5,), (0.1, 0.5, 0.9)):
        ours = _call(maz.ess, values, method="local", prob=prob)
        upstream = _call(az.ess, values, method="local", prob=prob)
        assert ours[0] == upstream[0] == "error", (prob, ours, upstream)
        assert ours[2] is upstream[2] is ValueError, (prob, ours[2], upstream[2])


def test_invalid_method_errors_match_upstream():
    values = np.random.default_rng(120).normal(size=(4, 20))
    for function in ("ess", "rhat"):
        ours = _call(getattr(maz, function), values, method="unknown")
        upstream = _call(getattr(az, function), values, method="unknown")
        assert ours[0] == upstream[0] == "error", (function, ours, upstream)
        assert ours[2] is upstream[2] is ValueError, (function, ours[2], upstream[2])


# ---------------------------------------------------------------------------
# xarray containers: attrs, coordinates, variable selection, dim rules.
# ---------------------------------------------------------------------------


def test_dataarray_attrs_and_coords_match_upstream():
    values = np.random.default_rng(121).normal(size=(4, 30, 3))
    array = xr.DataArray(
        values,
        dims=("chain", "draw", "p"),
        coords={"p": ["a", "b", "c"]},
        name="theta",
        attrs={"units": "m"},
    )
    for function in ("ess", "rhat"):
        ours = getattr(maz, function)(array)
        theirs = getattr(az, function)(array)
        assert ours.dims == theirs.dims
        assert ours.name == theirs.name
        assert ours.attrs == theirs.attrs
        for dim in theirs.dims:
            np.testing.assert_array_equal(np.asarray(ours.coords[dim]), np.asarray(theirs.coords[dim]))
        assert_close(ours, theirs, atol=2e-10)

    ours_hdi = maz.hdi(array, prob=0.6)
    theirs_hdi = az.hdi(array, prob=0.6)
    assert ours_hdi.dims == theirs_hdi.dims
    assert ours_hdi.name == theirs_hdi.name
    assert ours_hdi.attrs == theirs_hdi.attrs
    assert ours_hdi.ci_bound.attrs == theirs_hdi.ci_bound.attrs
    assert list(ours_hdi.ci_bound.values) == list(theirs_hdi.ci_bound.values)
    assert_close(ours_hdi, theirs_hdi, atol=0)


@pytest.mark.parametrize("dim", [None, "draw", ["chain", "draw"]])
def test_dataarray_hdi_dim_selection_matches_upstream(dim):
    values = np.random.default_rng(122).normal(size=(4, 21, 2))
    array = xr.DataArray(values, dims=("chain", "draw", "p"))
    ours = maz.hdi(array, prob=0.7, dim=dim)
    theirs = az.hdi(array, prob=0.7, dim=dim)
    assert ours.dims == theirs.dims
    assert_close(ours, theirs, atol=0)


@pytest.mark.parametrize("sample_dims", [["chain", "draw", "p"]])
def test_dataarray_bad_sample_dims_match_upstream(sample_dims):
    values = np.random.default_rng(123).normal(size=(4, 30, 2))
    array = xr.DataArray(values, dims=("chain", "draw", "p"))
    for function in ("ess", "rhat"):
        ours = _call(getattr(maz, function), array, sample_dims=sample_dims)
        upstream = _call(getattr(az, function), array, sample_dims=sample_dims)
        assert ours[0] == upstream[0] == "error", (sample_dims, function, ours, upstream)
        assert ours[2] is upstream[2] is ValueError, (sample_dims, function, ours[2], upstream[2])


def test_dataarray_single_dim_rhat_rejected_like_upstream():
    """upstream: arviz_stats.base.dataarray _DataArrayStats.rhat requires len(dims) == 2."""
    values = np.random.default_rng(127).normal(size=(4, 30, 2))
    array = xr.DataArray(values, dims=("chain", "draw", "p"))
    ours = _call(maz.rhat, array, sample_dims=["draw"])
    upstream = _call(az.rhat, array, sample_dims=["draw"])
    assert ours[0] == upstream[0] == "error", (ours, upstream)
    assert ours[2] is upstream[2] is ValueError, (ours[2], upstream[2])

    # ess accepts a single sample dim on both sides.
    assert_close(maz.ess(array, sample_dims=["draw"]), az.ess(array, sample_dims=["draw"]), atol=1e-9)


def test_dataset_variable_selection_matches_upstream():
    def fresh():
        rng = np.random.default_rng(124)
        return xr.Dataset(
            {
                "theta": (("chain", "draw"), rng.normal(size=(4, 30)), {"units": "m"}),
                "theta_aux": (("chain", "draw"), rng.normal(size=(4, 30))),
                "mu": (("chain", "draw"), rng.normal(size=(4, 30))),
            }
        )

    selections = [
        {"var_names": ["mu", "theta"]},
        {"var_names": ["theta"], "filter_vars": "like"},
        {"var_names": ["^theta"], "filter_vars": "regex"},
        {"var_names": ["~theta"]},
        {"var_names": ["~theta"], "filter_vars": "regex"},
        {"var_names": ["~does_not_exist"]},
        {"var_names": "mu"},
        {"var_names": ["missing"]},
    ]
    for selection in selections:
        ours = _call(maz.ess, fresh(), **selection)
        upstream = _call(az.ess, fresh(), **selection)
        if upstream[0] == "error":
            _assert_same_error(ours, upstream)
            continue
        assert ours[0] == "ok", (selection, ours)
        assert list(ours[1].data_vars) == list(upstream[1].data_vars), selection
        for name in upstream[1]:
            assert_close(ours[1][name], upstream[1][name], atol=1e-9)


def test_dataset_negation_with_like_or_regex_matches_upstream():
    """upstream: arviz_base.utils._subset_list applies "~name" exclusions, then filters.

    ArviZ itself raises TypeError when the excluded name exists *and* a like/regex
    filter is combined with it, so that narrower form is not pinned here.
    """

    def fresh():
        rng = np.random.default_rng(128)
        return xr.Dataset(
            {
                "theta": (("chain", "draw"), rng.normal(size=(4, 30))),
                "theta_aux": (("chain", "draw"), rng.normal(size=(4, 30))),
                "mu": (("chain", "draw"), rng.normal(size=(4, 30))),
            }
        )

    # Values recorded from az.ess(...) on the same fixture.
    ours = maz.ess(fresh(), var_names=["~theta_aux"], filter_vars="like")
    assert list(ours.data_vars) == ["theta", "mu"]
    ours = maz.ess(fresh(), var_names=["~theta"], filter_vars="regex")
    assert list(ours.data_vars) == ["mu"]
    theirs = az.ess(fresh(), var_names=["~theta"], filter_vars="regex")
    assert list(theirs.data_vars) == ["mu"]
    for name in ours.data_vars:
        assert_close(ours[name], az.ess(fresh(), var_names=["~theta"], filter_vars="regex")[name], atol=1e-9)


def test_dataset_unsupported_filter_vars_matches_upstream():
    rng = np.random.default_rng(129)
    dataset = xr.Dataset({"mu": (("chain", "draw"), rng.normal(size=(4, 30)))})
    ours = _call(maz.ess, dataset, var_names=["mu"], filter_vars="bogus")
    upstream = _call(az.ess, dataset, var_names=["mu"], filter_vars="bogus")
    assert ours[0] == upstream[0] == "error", (ours, upstream)
    assert ours[2] is upstream[2] is ValueError, (ours[2], upstream[2])


def test_dataset_group_and_attrs_match_upstream():
    rng = np.random.default_rng(125)
    dataset = xr.Dataset(
        {
            "theta": (("chain", "draw"), rng.normal(size=(4, 30)), {"units": "m"}),
            "mu": (("chain", "draw"), rng.normal(size=(4, 30))),
        }
    )
    ours, theirs = maz.ess(dataset), az.ess(dataset)
    assert list(ours.data_vars) == list(theirs.data_vars)
    for name in theirs:
        assert_close(ours[name], theirs[name], atol=1e-9)
        assert ours[name].attrs == theirs[name].attrs

    ours_prior = _call(maz.ess, dataset, group="prior")
    theirs_prior = _call(az.ess, dataset, group="prior")
    assert ours_prior[0] == theirs_prior[0] == "ok"
    assert_close(ours_prior[1]["theta"], theirs_prior[1]["theta"], atol=1e-9)


def test_dataset_nan_and_duplicates_match_upstream():
    rng = np.random.default_rng(126)
    values = np.round(rng.normal(size=(3, 40)), 0)
    values[0, 2] = np.nan
    dataset = xr.Dataset({"x": (("chain", "draw"), values)})
    assert np.isnan(maz.ess(dataset)["x"])
    assert np.isnan(az.ess(dataset)["x"])
    clean = xr.Dataset({"x": (("chain", "draw"), values[:, 1:])})
    for function in ("ess", "rhat"):
        assert_close(getattr(maz, function)(clean)["x"], getattr(az, function)(clean)["x"], atol=2e-10)


# ---------------------------------------------------------------------------
# SIMD paths: widths 0, 1, W-1, W, W+1 and a large odd length so the scalar tails
# and the partial trailing blocks are exercised.
#
# The rhat kernel and the rank prepare/tie loops are vectorised; both are also
# reached through the public API. W is read from the compiled kernel by building
# a two-line shared library, so the tests follow the host instead of assuming 4.
# MOJOARVIZ_SIMD_WIDTH widens the same cases further (e.g. 8 or 16) to prove the
# scalar tail at a larger W; the arithmetic is identical, only the block size
# changes.
# ---------------------------------------------------------------------------

_WIDTH_SOURCE = """from std.sys.info import simd_width_of

@export("mav_probe_simd_width")
def mav_probe_simd_width() abi("C") -> Int:
    return simd_width_of[DType.float64]()
"""


def _probe_simd_width() -> int:
    override = os.environ.get("MOJOARVIZ_SIMD_WIDTH")
    if override:
        return int(override)
    root = Path(__file__).resolve().parents[1]
    mojo = os.environ.get("MOJO_BIN") or shutil.which("mojo")
    if mojo is None:
        candidate = root / ".pixi" / "envs" / "default" / "bin" / "mojo"
        mojo = str(candidate) if candidate.exists() else None
    if mojo is None:
        return 4
    env = dict(os.environ)
    env.setdefault("MODULAR_HOME", str(Path(mojo).resolve().parents[1] / "share" / "max"))
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "probe.mojo"
        source.write_text(_WIDTH_SOURCE)
        library = Path(scratch) / "probe.so"
        try:
            proc = subprocess.run(
                [mojo, "build", "--emit", "shared-lib", str(source), "-o", str(library)],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
            if proc.returncode:
                return 4
            probe = ctypes.CDLL(str(library))
            probe.mav_probe_simd_width.restype = ctypes.c_int64
            return int(probe.mav_probe_simd_width())
        except (OSError, subprocess.TimeoutExpired, AttributeError):
            return 4


WIDTH = _probe_simd_width()
SIMD_LENGTHS = [0, 1, WIDTH - 1, WIDTH, WIDTH + 1, 2 * WIDTH + 3, 200_003]


@pytest.mark.parametrize("draw_count", SIMD_LENGTHS)
def test_core_rhat_simd_tails(draw_count):
    """Both rhat passes, including each scalar tail and an odd row start."""
    rng = np.random.default_rng(200 + draw_count)
    chains = 4
    values = rng.normal(size=(chains, draw_count))
    # the public entry point: degenerate sizes must agree on "raise" vs "value"
    ours = _call(maz.rhat, values, method="identity")
    upstream = _call(az.rhat, values, method="identity")
    assert ours[0] == upstream[0], (draw_count, ours, upstream)
    if upstream[0] == "ok":
        assert_close(ours[1], upstream[1], atol=2e-10)
    if draw_count < 2:
        # core_rhat rejects fewer than two draws, like upstream's validity gate
        return
    # core_rhat against the closed form, computed in float64 directly
    chain_means = values.mean(axis=1)
    within = values.var(axis=1, ddof=1).mean()
    between = draw_count * chain_means.var(ddof=1)
    assert_close(core_rhat(values), np.sqrt((between / within + draw_count - 1) / draw_count), atol=1e-12)
    if draw_count >= 4:
        assert_close(core_rhat(values), _DiagnosticsBase()._rhat(values), atol=1e-12)


@pytest.mark.parametrize("draws", [200_001, 200_002, 200_003])
def test_core_rhat_unaligned_chain_starts(draws):
    """Chain rows of 200,001+ draws do not start on a 4-element block."""
    rng = np.random.default_rng(250 + draws)
    values = rng.normal(size=(4, draws))
    assert_close(core_rhat(values), _DiagnosticsBase()._rhat(values), atol=1e-12)
    # 1,000,001 draws: a row is one draw off a 4-element block, so a 3-chain
    # buffer shifts each row's start by a different phase.
    wide = rng.normal(size=(3, 1_000_001))
    assert_close(core_rhat(wide), _DiagnosticsBase()._rhat(wide), atol=1e-12)


@pytest.mark.parametrize("size", SIMD_LENGTHS)
def test_rank_normalize_simd_tails(size, upstream_base):
    """Rank prepare, quicksort hand-off and the tie loop at every tail width."""
    _, diagnostics = upstream_base
    if size == 0:
        with pytest.raises(ValueError):
            rank_normalize(np.empty((1, 0)))
        return
    rng = np.random.default_rng(400 + size)
    distinct = rng.normal(size=size)
    tied = np.round(rng.normal(size=size), 1)
    if size >= 3:
        # one run of ties that starts mid-block and ends at the last value
        tied[-3:] = tied[0]
    for case in (distinct, tied):
        assert_close(rank_normalize(case), diagnostics._z_scale(case), atol=2e-8)


def test_rank_normalize_long_tied_runs(upstream_base):
    """Tie runs longer than a vector: the scatter is scalar, the value is not."""
    _, diagnostics = upstream_base
    for size in (WIDTH * 4, 1_003, 200_003):
        values = np.repeat(np.arange(size // 7 + 1, dtype=np.float64), 7)[:size]
        assert_close(rank_normalize(values), diagnostics._z_scale(values), atol=2e-8)
