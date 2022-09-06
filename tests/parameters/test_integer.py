# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import scipy.stats as sps

from parameterspace.parameters.base import BaseParameter
from parameterspace.parameters.integer import IntegerParameter
from parameterspace.transformations.log_zero_one import LogZeroOneInteger

from .util import check_sampling, check_value_numvalue_conversion, check_values


def test_integer_parameter():
    bounds = np.array((-4, 8))
    p = IntegerParameter("foo", bounds)

    assert p.is_continuous is False
    assert p.is_ordered

    check_values(p, [-5, 2, 2.1, 5], [False, True, False, True])
    check_value_numvalue_conversion(
        p, [(i + bounds[0], (i + 0.5) / (bounds[1] - bounds[0] + 1)) for i in range(13)]
    )
    check_sampling(p)

    # test string representation
    str(p)


@pytest.mark.flaky(max_runs=4)
def test_integer_log_transform(num_samples=2**14):
    bounds = np.array((1, 32))
    p = IntegerParameter("bar", bounds, transformation="log")

    assert not p.is_continuous
    assert p.is_ordered
    assert isinstance(p._transformation, LogZeroOneInteger)

    check_values(p, [0, 2, 2.1, 5], [False, True, False, True])
    check_sampling(p)

    samples = p.sample_values(num_samples=num_samples)

    # uniform distribution in log space is equivalent to the reciprocal distribution
    # reference: https://en.wikipedia.org/wiki/Reciprocal_distribution
    sps_dist = sps.reciprocal(bounds[0] - 0.5, bounds[1] + 0.5)
    expected_frequencies = len(samples) * np.array(
        [
            (sps_dist.cdf(i + 0.5) - sps_dist.cdf(i - 0.5))
            for i in np.arange(bounds[0], bounds[1] + 1)
        ]
    )
    res = sps.chisquare(np.bincount(samples)[1:], expected_frequencies)
    chi2 = sps.chi2(bounds[1] - bounds[0])
    assert chi2.cdf(res.statistic) < 0.95


@pytest.mark.flaky(max_runs=4)
def test_integer_uniformity(num_samples=2**14):
    bounds = np.array((-4, 8), dtype=int)
    p1 = IntegerParameter("bar", bounds)
    samples1 = p1.sample_values(num_samples)

    counts1 = np.bincount(np.array(samples1) - bounds[0])
    num_values1 = 1 + bounds[1] - bounds[0]

    res = sps.chisquare(counts1)
    chi2 = sps.chi2(num_values1 - 1)
    assert chi2.cdf(res.statistic) < 0.95

    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BaseParameter.from_dict(json_dict)
    assert p1 == p2

    samples2 = p2.sample_values(num_samples)

    counts2 = np.bincount(np.array(samples2) - bounds[0])
    num_values2 = 1 + bounds[1] - bounds[0]

    res = sps.chisquare(counts2)
    chi2 = sps.chi2(num_values2 - 1)
    assert chi2.cdf(res.statistic) < 0.95
