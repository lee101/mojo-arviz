"""Array and xarray interfaces matching ArviZ's posterior diagnostics."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from ._lib import (
    core_ess,
    core_rhat,
    hdi_nearest,
    rank_normalize,
    rank_normalize_pair,
)

_ESS_METHODS = {
    "bulk",
    "tail",
    "mean",
    "sd",
    "quantile",
    "local",
    "median",
    "mad",
    "z_scale",
    "folded",
    "identity",
}
_RHAT_METHODS = {"rank", "folded", "z_scale", "split", "identity"}


def _sample_array(data, chain_axis, draw_axis):
    array = np.asarray(data)
    if chain_axis is None:
        array = np.expand_dims(array, axis=0)
        chain_axis = 0
        if draw_axis > 0:
            draw_axis += 1
    moved = np.moveaxis(array, (chain_axis, draw_axis), (-2, -1))
    return np.ascontiguousarray(moved, dtype=np.float64)


def _split_chains(values):
    half = values.shape[1] // 2
    return np.ascontiguousarray(
        np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    )


def _valid(values, min_chains):
    return values.shape[0] >= min_chains and values.shape[1] >= 4 and np.isfinite(values).all()


def _ess_one(values, method, relative, prob):
    if not _valid(values, 1):
        return np.nan
    if method == "identity":
        transformed = values
    elif method == "mean":
        transformed = _split_chains(values)
    elif method in {"bulk", "z_scale"}:
        transformed = rank_normalize(_split_chains(values))
    elif method == "folded":
        split = _split_chains(values)
        transformed = rank_normalize(np.abs(split - np.median(split)))
    elif method == "sd":
        transformed = _split_chains((values - values.mean()) ** 2)
    elif method in {"quantile", "median"}:
        if method == "quantile" and prob is None:
            raise TypeError("Prob not defined.")
        quantile = 0.5 if method == "median" else prob
        threshold = np.quantile(values, quantile)
        transformed = _split_chains(values <= threshold)
    elif method == "local":
        if prob is None:
            raise TypeError("Prob not defined.")
        if len(prob) != 2:
            raise ValueError("Prob argument in ess local must be upper and lower bound")
        lower, upper = np.quantile(values, prob)
        transformed = _split_chains((lower <= values) & (values <= upper))
    elif method == "tail":
        if prob is None:
            raise TypeError("_ess_tail() missing required argument: 'prob'")
        bounds = sorted((prob, 1 - prob)) if not isinstance(prob, Sequence) else prob
        return min(
            _ess_one(values, "quantile", relative, bounds[0]),
            _ess_one(values, "quantile", relative, bounds[1]),
        )
    elif method == "mad":
        folded = np.abs(values - np.median(values))
        indicator = folded <= np.median(folded)
        transformed = rank_normalize(_split_chains(indicator))
    else:
        raise AssertionError(method)
    return core_ess(np.asarray(transformed, dtype=np.float64), relative)


def _rhat_one(values, method):
    if not _valid(values, 2):
        return np.nan
    if method == "identity":
        return core_rhat(values)
    split = _split_chains(values)
    if method == "split":
        return core_rhat(split)
    if method == "z_scale":
        return core_rhat(rank_normalize(split))
    if method == "folded":
        return core_rhat(rank_normalize(np.abs(split - np.median(split))))
    bulk_values, folded_values = rank_normalize_pair(
        split,
        np.abs(split - np.median(split)),
    )
    bulk = core_rhat(bulk_values)
    folded = core_rhat(folded_values)
    return max(bulk, folded)


def _map_samples(data, fn, chain_axis, draw_axis):
    array = _sample_array(data, chain_axis, draw_axis)
    batch_shape = array.shape[:-2]
    batch = array.reshape((-1,) + array.shape[-2:])
    result = np.empty(batch.shape[0], dtype=np.float64)
    for index, values in enumerate(batch):
        result[index] = fn(values)
    return result.reshape(batch_shape)


def _select_vars(dataset, var_names, filter_vars):
    if var_names is None:
        return list(dataset.data_vars)
    names = [var_names] if isinstance(var_names, str) else list(var_names)
    if filter_vars is None:
        return names
    if filter_vars == "like":
        return [name for name in dataset.data_vars if any(term in name for term in names)]
    if filter_vars == "regex":
        return [name for name in dataset.data_vars if any(re.search(term, name) for term in names)]
    raise ValueError("filter_vars must be one of None, 'like', or 'regex'")


def _container_group(data, group):
    try:
        import xarray as xr
    except ImportError:
        return None, None
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        return data, xr
    if hasattr(data, group):
        return getattr(data, group), xr
    try:
        child = data[group]
        return getattr(child, "dataset", child), xr
    except (KeyError, TypeError):
        return None, xr


def _sample_dims(data_array, sample_dims):
    if sample_dims is None:
        sample_dims = [dim for dim in ("chain", "draw") if dim in data_array.dims]
    elif isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    else:
        sample_dims = list(sample_dims)
    if not sample_dims:
        raise ValueError("no sample dimensions found")
    return sample_dims


def _apply_xarray_diagnostic(data, function, sample_dims, group, var_names, filter_vars, coords):
    container, xr = _container_group(data, group)
    if container is None:
        return None
    if coords is not None:
        container = container.sel(coords)

    def apply_one(array):
        dims = _sample_dims(array, sample_dims)
        if len(dims) > 2:
            raise ValueError("sample_dims can only have 1 or 2 elements")
        remaining = [dim for dim in array.dims if dim not in dims]
        ordered = array.transpose(*(remaining + dims))
        chain_axis = -2 if len(dims) == 2 else None
        draw_axis = -1
        values = function(ordered.values, chain_axis, draw_axis)
        coords_out = {dim: ordered.coords[dim] for dim in remaining}
        return xr.DataArray(values, dims=remaining, coords=coords_out, name=array.name)

    if isinstance(container, xr.DataArray):
        return apply_one(container)
    selected = _select_vars(container, var_names, filter_vars)
    return xr.Dataset({name: apply_one(container[name]) for name in selected})


def ess(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="bulk",
    relative=False,
    prob=None,
    chain_axis=0,
    draw_axis=1,
):
    """Estimate effective sample size with ArviZ-compatible methods and axes."""
    method = method.lower()
    if method not in _ESS_METHODS:
        raise ValueError(f"Requested method '{method}' but it must be one of {_ESS_METHODS}")
    if not isinstance(data, (list, tuple, np.ndarray)):
        result = _apply_xarray_diagnostic(
            data,
            lambda ary, ca, da: _map_samples(
                ary, lambda values: _ess_one(values, method, relative, prob), ca, da
            ),
            sample_dims,
            group,
            var_names,
            filter_vars,
            coords,
        )
        if result is not None:
            return result
    return _map_samples(
        data, lambda values: _ess_one(values, method, relative, prob), chain_axis, draw_axis
    )


def rhat(
    data,
    sample_dims=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="rank",
    chain_axis=0,
    draw_axis=1,
):
    """Compute rank-normalized, folded, split, or identity R-hat."""
    method = method.lower()
    if method not in _RHAT_METHODS:
        raise ValueError(f"Requested method '{method}' but it must be one of {_RHAT_METHODS}")
    if not isinstance(data, (list, tuple, np.ndarray)):
        result = _apply_xarray_diagnostic(
            data,
            lambda ary, ca, da: _map_samples(
                ary, lambda values: _rhat_one(values, method), ca, da
            ),
            sample_dims,
            group,
            var_names,
            filter_vars,
            coords,
        )
        if result is not None:
            return result
    return _map_samples(data, lambda values: _rhat_one(values, method), chain_axis, draw_axis)


def _hdi_python(values, prob):
    sorted_values = np.sort(values)
    interval_idx = int(np.floor(prob * len(sorted_values)))
    widths = sorted_values[interval_idx:] - sorted_values[: len(sorted_values) - interval_idx]
    if not len(widths):
        raise ValueError("Too few elements for interval calculation. ")
    best = np.argmin(widths)
    return sorted_values[[best, best + interval_idx]]


def _standardize_angle(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


def _hdi_one(values, prob, circular, skipna):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if skipna:
        mask = np.isnan(values)
        if not mask.all():
            values = values[~mask]
    if circular:
        mean = np.angle(np.mean(np.exp(1j * values)))
        values = _standardize_angle(values - mean)
    interval_idx = int(np.floor(prob * len(values)))
    if interval_idx >= len(values):
        raise ValueError("Too few elements for interval calculation. ")
    if np.isnan(values).any():
        interval = _hdi_python(values, prob)
    else:
        interval = hdi_nearest(values, interval_idx)
    return _standardize_angle(interval + mean) if circular else interval


def _hdi_array(data, prob, axis, circular, skipna):
    array = np.asarray(data)
    if axis is None:
        axes = tuple(range(array.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    axes = tuple(ax if ax >= 0 else array.ndim + ax for ax in axes)
    remaining = tuple(index for index in range(array.ndim) if index not in axes)
    moved = np.transpose(array, remaining + axes)
    reduce_size = int(np.prod([array.shape[axis] for axis in axes], dtype=np.int64))
    batch_shape = moved.shape[: len(remaining)]
    batch = moved.reshape((-1, reduce_size))
    result = np.empty((batch.shape[0], 2), dtype=np.float64)
    for index, values in enumerate(batch):
        result[index] = _hdi_one(values, prob, circular, skipna)
    return result.reshape(batch_shape + (2,))


def hdi(
    data,
    prob=None,
    dim=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    method="nearest",
    circular=False,
    max_modes=10,
    skipna=False,
    **kwargs,
):
    """Compute the shortest sample interval containing ``prob`` mass."""
    del max_modes
    prob = 0.89 if prob is None else prob
    if not 1 >= prob > 0:
        raise ValueError("The value of `prob` must be in the (0, 1] interval.")
    if method != "nearest":
        raise NotImplementedError("mojoarviz currently covers hdi(method='nearest') only")
    if isinstance(data, (list, tuple, np.ndarray)):
        axis = kwargs.pop("axis", -1)
        if kwargs:
            key = next(iter(kwargs))
            raise TypeError(f"hdi got an unexpected keyword argument: '{key}'")
        return _hdi_array(data, prob, axis, circular, skipna)

    container, xr = _container_group(data, group)
    if container is None:
        return _hdi_array(data, prob, kwargs.pop("axis", -1), circular, skipna)
    if coords is not None:
        container = container.sel(coords)

    def apply_one(array):
        dims = _sample_dims(array, dim)
        remaining = [name for name in array.dims if name not in dims]
        ordered = array.transpose(*(remaining + dims))
        axes = tuple(range(-len(dims), 0))
        values = _hdi_array(ordered.values, prob, axes, circular, skipna)
        coords_out = {name: ordered.coords[name] for name in remaining}
        coords_out["ci_bound"] = ["lower", "upper"]
        return xr.DataArray(
            values,
            dims=remaining + ["ci_bound"],
            coords=coords_out,
            name=array.name,
        )

    if isinstance(container, xr.DataArray):
        return apply_one(container)
    selected = _select_vars(container, var_names, filter_vars)
    return xr.Dataset({name: apply_one(container[name]) for name in selected})
