"""Numerical and behavioral parity against the installed ArviZ release."""

from __future__ import annotations

import inspect

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
