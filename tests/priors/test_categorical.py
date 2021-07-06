# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json

import numpy as np
import pytest

from parameterspace.priors.base import BasePrior
from parameterspace.priors.categorical import Categorical

PRIOR_PROBABILITIES = [1, 2, 1, 3, 5]


def test_categorical_prior_bounds():
    p = Categorical(PRIOR_PROBABILITIES)

    assert np.allclose(p.bounds, [0, 1], 1e-6)


def test_categorical_prior_to_string():
    p = Categorical(PRIOR_PROBABILITIES)

    try:
        assert "Categorical " in str(p)
    except:
        assert False


def test_categorical_prior_sampling(num_samples=64):
    p = Categorical(PRIOR_PROBABILITIES)
    samples = p.sample(num_samples)

    assert samples.shape == (num_samples,)
    assert np.all(samples >= 0)
    assert np.all(samples < 1)


def test_categorical_prior_sampling_deterministic_seed(num_samples=64):
    p = Categorical(PRIOR_PROBABILITIES)
    rng = np.random.RandomState(42)
    samples1 = p.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


def test_categorical_prior_pdf_and_likelihood(num_samples=64):
    p = Categorical(PRIOR_PROBABILITIES)
    samples = p.sample(num_samples)
    lls = p.loglikelihood(samples)

    Z = np.sum(PRIOR_PROBABILITIES)

    for s in samples:
        assert (
            p.pdf(s)
            == PRIOR_PROBABILITIES[int(np.around(len(PRIOR_PROBABILITIES) * s - 0.5))]
            / Z
        )

    for i, j in itertools.combinations(range(len(lls)), 2):
        ll_diff1 = lls[i] - lls[j]
        ll_diff2 = np.log(
            PRIOR_PROBABILITIES[
                int(np.around(len(PRIOR_PROBABILITIES) * samples[i] - 0.5))
            ]
            / PRIOR_PROBABILITIES[
                int(np.around(len(PRIOR_PROBABILITIES) * samples[j] - 0.5))
            ]
        )
        assert np.allclose(ll_diff1, ll_diff2, 1e-6)


def test_categorical_prior_pdf_and_likelihood_unknown_value(num_samples=64):
    p = Categorical(PRIOR_PROBABILITIES)
    samples = p.sample(num_samples)
    samples[0] = len(PRIOR_PROBABILITIES)

    with pytest.raises(ValueError):
        assert np.all(np.isnan(p.pdf(samples)))


def test_categorical_prior_pdf_and_likelihood_nans_infs(num_samples=64):
    p = Categorical(PRIOR_PROBABILITIES)
    samples = np.full(num_samples, np.nan)
    samples[::2] = np.inf

    assert np.all(np.isnan(p.pdf(samples)))


def test_categorical_prior_invalid_PRIOR_PROBABILITIES():
    PRIOR_PROBABILITIES = [1, 2, 1, 3, -5]

    with pytest.raises(ValueError):
        p = Categorical(PRIOR_PROBABILITIES)


def test_categorical_prior_to_from_dict(num_samples=64):
    p1 = Categorical(PRIOR_PROBABILITIES)
    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BasePrior.from_dict(json_dict)

    rng = np.random.RandomState(42)
    samples1 = p1.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p2.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
