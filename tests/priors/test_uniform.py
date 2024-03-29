# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from parameterspace.priors.base import BasePrior
from parameterspace.priors.uniform import Uniform


def test_uniform_prior_bounds():
    p = Uniform()
    assert np.allclose(p.bounds, [0, 1], 1e-6)


def test_uniform_prior_to_string():
    p = Uniform()

    try:
        assert "Uniform" in str(p)
    except:  # pylint: disable=bare-except
        assert False


def test_uniform_float_prior_sampling(num_samples=64):
    p = Uniform()
    samples = p.sample(num_samples)

    assert samples.shape == (num_samples,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


def test_uniform_prior_sampling_deterministic_seed(num_samples=64):
    p = Uniform()
    rng = np.random.RandomState(42)
    samples1 = p.sample(num_samples, random_state=rng)
    rng.seed(42)
    samples2 = p.sample(num_samples, random_state=rng)

    assert np.all(samples1 == samples2)


def test_uniform_prior_pdf_and_likelihood_within_bounds(num_samples=64):
    p = Uniform()
    samples = p.sample(num_samples)
    lls = p.loglikelihood(samples)

    assert np.allclose(1, p.pdf(samples))
    assert np.allclose(lls, lls[0], 1e-6)


def test_uniform_prior_pdf_and_likelihood_outside_bounds(num_samples=64):
    p = Uniform()
    samples = p.sample(num_samples) + 1
    lls = p.loglikelihood(samples)

    assert np.allclose(0, p.pdf(samples))
    assert np.all(np.isinf(lls))


def test_uniform_prior_pdf_and_likelihood_nans(num_samples=64):
    p = Uniform()
    samples = np.full(num_samples, np.nan)

    assert np.all(np.isnan(p.pdf(samples)))


def test_uniform_prior_to_from_dict(num_samples=64):
    p1 = Uniform()
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
