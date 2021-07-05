# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import scipy.stats as sps

from parameterspace.parameters.base import BaseParameter
from parameterspace.parameters.categorical import CategoricalParameter
from parameterspace.parameters.continuous import ContinuousParameter
from parameterspace.parameters.integer import IntegerParameter
from parameterspace.priors.uniform import Uniform
from parameterspace.transformations.log_zero_one import LogZeroOneFloat
from parameterspace.transformations.zero_one import ZeroOneFloat, ZeroOneInteger

from .util import check_sampling, check_value_numvalue_conversion, check_values


@pytest.mark.flaky(max_runs=4)
def test_continuous_parameter(num_samples=2 ** 14):
    bounds = np.array((-8, 4))
    p = ContinuousParameter("foo", bounds)

    assert p.is_continuous
    assert p.is_ordered

    check_values(p, [-5, 2, 2.1, 5], [True, True, True, False])
    check_value_numvalue_conversion(
        p,
        [
            (f, (f - bounds[0]) / (bounds[1] - bounds[0]))
            for f in np.linspace(bounds[0], bounds[1], 128)
        ],
    )
    check_sampling(p)

    samples = p.sample_values(num_samples=num_samples)
    stat, p_value = sps.kstest(
        samples, sps.uniform(loc=bounds[0], scale=(bounds[1] - bounds[0])).cdf
    )
    # KS statistic should be less than this value for confidence alpha=0.05
    # reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    assert stat * np.sqrt(num_samples) < 1.36
    # test string representation
    str(p)


@pytest.mark.flaky(max_runs=4)
def test_continuous_log_transform(num_samples=2 ** 14):
    bounds = np.array((0.004, 4))
    p1 = ContinuousParameter("bar", bounds, transformation="log")

    assert p1.is_continuous
    assert p1.is_ordered
    assert isinstance(p1._transformation, LogZeroOneFloat)

    check_values(p1, [0, 2, 2.1, 5], [False, True, True, False])
    check_sampling(p1)

    samples = p1.sample_values(num_samples=num_samples)
    # uniform distribution in log space is equivalent to the reciprocal distribution
    # reference: https://en.wikipedia.org/wiki/Reciprocal_distribution
    stat, p_value = sps.kstest(samples, sps.reciprocal(bounds[0], bounds[1]).cdf)
    # KS statistic should be less than this value for confidence alpha=0.05
    # reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    assert stat * np.sqrt(num_samples) < 1.36

    # test string representation
    str(p1)

    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BaseParameter.from_dict(json_dict)
    assert p1 == p2

    samples = p2.sample_values(num_samples=num_samples)
    # uniform distribution in log space is equivalent to the reciprocal distribution
    # reference: https://en.wikipedia.org/wiki/Reciprocal_distribution
    stat, p_value = sps.kstest(samples, sps.reciprocal(bounds[0], bounds[1]).cdf)
    # KS statistic should be less than this value for confidence alpha=0.05
    # reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    assert stat * np.sqrt(num_samples) < 1.36
