# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np

from parameterspace.priors.normal import Normal

mean, std = 1, 5


def test_normal_prior_bounds():
    p = Normal(mean, std)

    assert np.all(np.isinf(p.bounds))


def test_normal_prior_to_string():
    p = Normal(mean, std)

    try:
        assert "Normal" in str(p)
    except:
        assert False


def test_normal_prior_sampling(num_samples=64):
    p = Normal(mean, std)
    samples = p.sample(num_samples)

    assert samples.shape == (num_samples,)
    assert np.all(np.isfinite(samples))


def test_normal_prior_sampling_deterministic_seed(num_samples=64):
    p = Normal(mean, std)
    rng = np.random.RandomState(42)
    samples1 = p.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


def test_normal_prior_pdf_and_likelihood_within_bounds(num_samples=64):
    p = Normal(mean, std)
    samples = p.sample(num_samples)
    lls = p.loglikelihood(samples)

    poor_mans_normal = lambda x: np.exp(-np.power(x - mean, 2) / (2 * np.power(std, 2)))
    pdfs = poor_mans_normal(samples) / np.sqrt(2 * np.pi) / std

    assert np.allclose(pdfs, p.pdf(samples))

    for i, j in itertools.combinations(range(len(lls)), 2):
        ll_diff1 = lls[i] - lls[j]
        ll_diff2 = np.log(poor_mans_normal(samples[i]) / poor_mans_normal(samples[j]))

        assert np.allclose(ll_diff1, ll_diff2, 1e-6)


def test_normal_prior_pdf_and_likelihood_nans(num_samples=64):
    p = Normal(mean, std)
    samples = np.full(num_samples, np.nan)

    assert np.all(np.isnan(p.pdf(samples)))
