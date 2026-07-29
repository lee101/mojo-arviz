"""Compute diagnostics for four autocorrelated chains."""

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
