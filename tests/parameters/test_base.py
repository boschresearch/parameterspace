# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from parameterspace.parameters.base import BaseParameter
from parameterspace.transformations.zero_one import ZeroOneFloat
from parameterspace.priors.uniform import Uniform

from .util import check_value_numvalue_conversion, check_sampling


class DummyParameter(BaseParameter):
    def check_value(self):
        pass


def test_base_parameter():
    bounds = np.array((-4, 8))
    prior = Uniform()

    p = DummyParameter("foo", prior, transformation=ZeroOneFloat(bounds), is_continuous=False, is_ordered=False, num_values=np.inf)
    q = DummyParameter("bar", prior, transformation=ZeroOneFloat(bounds), is_continuous=False, is_ordered=False, num_values=np.inf)

    check_value_numvalue_conversion(p, [(f, (f - bounds[0]) / (bounds[1] - bounds[0])) for f in np.linspace(bounds[0], bounds[1], 128)])
    check_sampling(p)

    ref_ll = -np.log(bounds[1] - bounds[0])

    for v in [-4, 0, 4]:
        assert np.allclose(p.loglikelihood(v), ref_ll, 1e-6)
        assert np.allclose(p.loglikelihood_numerical_value(p.val2num(v)), ref_ll, 1e-6)

    for v in np.linspace(0, 1):
        assert np.allclose(p.pdf_numerical_value(v), 1.0 / (bounds[1] - bounds[0]), 1e-6)

    assert p != q
