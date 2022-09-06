# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json

import numpy as np
import pytest
import scipy.special as sps

from parameterspace.priors.base import BasePrior
from parameterspace.priors.beta import Beta

a, b = 2, 4


def test_beta_prior_bounds():
    p = Beta(a, b)

    assert np.allclose(p.bounds, [0, 1], 1e-6)


def test_beta_prior_to_string():
    p = Beta(a, b)

    try:
        assert "Beta" in str(p)
    except:  # pylint: disable=bare-except
        assert False


def test_beta_prior_sampling(num_samples=64):
    p = Beta(a, b)
    samples = p.sample(num_samples)

    assert samples.shape == (num_samples,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_beta_prior_sampling_deterministic_seed(num_samples=64):
    p = Beta(a, b)
    rng = np.random.RandomState(42)
    samples1 = p.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


def test_beta_prior_pdf_and_likelihood_within_bounds(num_samples=64):
    p = Beta(a, b)
    samples = p.sample(num_samples)
    lls = p.loglikelihood(samples)

    def _poor_mans_beta(x):
        return np.power(x, a - 1) * np.power(1 - x, b - 1)

    Z = sps.gamma(a) * sps.gamma(b) / sps.gamma(a + b)
    pdfs = _poor_mans_beta(samples) / Z

    assert np.allclose(pdfs, p.pdf(samples))

    for i, j in itertools.combinations(range(len(lls)), 2):
        ll_diff1 = lls[i] - lls[j]
        ll_diff2 = np.log(_poor_mans_beta(samples[i]) / _poor_mans_beta(samples[j]))

        assert np.allclose(ll_diff1, ll_diff2, 1e-6)


def test_beta_prior_pdf_and_likelihood_outside_bounds(num_samples=64):
    p = Beta(a, b)
    samples = p.sample(num_samples)
    samples += 1
    lls = p.loglikelihood(samples)
    assert np.allclose(0, p.pdf(samples))
    assert np.all(np.isinf(lls))


def test_beta_prior_pdf_and_likelihood_nans(num_samples=64):
    p = Beta(a, b)
    samples = np.full(num_samples, np.nan)
    assert np.all(np.isnan(p.pdf(samples)))


def test_beta_prior_sampling_deterministic_seed(num_samples=64):
    p1 = Beta(a, b)
    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BasePrior.from_dict(json_dict)
    assert p1 == p2

    rng = np.random.RandomState(42)
    samples1 = p2.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p1.sample(num_samples, random_state=rng)
    assert np.all(samples1 == samples2)


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
