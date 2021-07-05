# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json

import numpy as np
import scipy.special as sps

import pytest

from parameterspace.priors.truncated_normal import TruncatedNormal
from parameterspace.priors.base import BasePrior

mean, std = 1, 2


def test_truncated_normal_prior_bounds():
    p = TruncatedNormal(mean, std)
    assert np.allclose(p.bounds, [0, 1], 1e-6)


def test_truncated_normal_prior_to_string():
    p = TruncatedNormal(mean, std)

    try:
        assert "Truncated normal" in str(p)
    except:
        assert False


def test_truncated_normal_prior_sampling(num_samples=64):
    p = TruncatedNormal(mean, std)
    samples = p.sample(num_samples)

    assert samples.shape == (num_samples,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_truncated_normal_prior_sampling_deterministic_seed(num_samples=64):
    p = p = TruncatedNormal(mean, std)
    rng = np.random.RandomState(42)
    samples1 = p.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


def test_truncated_normal_prior_pdf_and_likelihood_within_bounds(num_samples=64):
    p = TruncatedNormal(mean, std)
    samples = p.sample(num_samples)
    lls = p.loglikelihood(samples)

    poor_mans_normal = lambda x: np.exp(-np.power(x - mean, 2) / (2 * np.power(std, 2)))
    # compute pdfs by hand
    Z = (sps.erf((1 - mean) / std / np.sqrt(2)) - sps.erf((0 - mean) / std / np.sqrt(2))) / 2
    pdfs = poor_mans_normal(samples) / Z / np.sqrt(2 * np.pi) / std

    assert np.allclose(pdfs, p.pdf(samples))

    for i, j in itertools.combinations(range(len(lls)), 2):
        ll_diff1 = lls[i] - lls[j]
        ll_diff2 = np.log(poor_mans_normal(samples[i]) / poor_mans_normal(samples[j]))

        assert np.allclose(ll_diff1, ll_diff2, 1e-6)


def test_truncated_normal_prior_pdf_and_likelihood_outside_bounds(num_samples=64):
    p = TruncatedNormal(mean, std)
    samples = p.sample(num_samples)
    samples += 1
    lls = p.loglikelihood(samples)

    assert np.allclose(0, p.pdf(samples))
    assert np.all(np.isinf(lls))


def test_truncated_normal_prior_pdf_and_likelihood_nans(num_samples=64):
    p = TruncatedNormal(mean, std)
    samples = np.full(num_samples, np.nan)

    assert np.all(np.isnan(p.pdf(samples)))


def test_truncated_normal_prior_to_from_dict(num_samples=64):
    p1 = TruncatedNormal(mean, std)
    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BasePrior.from_dict(json_dict)
    assert p1 == p2

    rng = np.random.RandomState(42)
    samples1 = p1.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p2.sample(num_samples, random_state=rng)
    assert np.all(samples1 == samples2)


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
