# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def check_sampling(p, num_samples=32):
    samples = p.sample_values(num_samples)
    assert len(samples) == num_samples

    bounds = p.get_numerical_bounds()
    samples = p.sample_numerical_values(num_samples)

    assert samples.shape[0] == num_samples
    assert np.all(np.array(samples) >= bounds[0])
    assert np.all(np.array(samples) <= bounds[1])


def check_value_numvalue_conversion(p, ref, num_values=True):

    for v, tv in ref:

        assert np.allclose(p.val2num(v), tv, 1e-6)

        if num_values:
            assert np.allclose(p.num2val(tv), v, 1e-6)
        else:
            assert p.num2val(tv) == v


def check_values(p, values, valid):

    for value, v in zip(values, valid):
        assert p.check_value(value) == v

        if v:  # only apply the transformation if the value is actually valid
            assert p.check_numerical_value(p.val2num(value)) == v
