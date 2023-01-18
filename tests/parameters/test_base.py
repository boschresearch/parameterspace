# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from parameterspace.parameters.base import BaseParameter
from parameterspace.priors.uniform import Uniform
from parameterspace.transformations.zero_one import ZeroOneFloat

from .util import check_sampling, check_value_numvalue_conversion


class PseudoParameter(BaseParameter):
    def check_value(self, value):
        pass


def test_base_parameter():
    bounds = np.array((-4, 8))
    prior = Uniform()

    p = PseudoParameter(
        "foo",
        prior,
        transformation=ZeroOneFloat(bounds),
        is_continuous=False,
        is_ordered=False,
        num_values=np.inf,
    )
    q = PseudoParameter(
        "bar",
        prior,
        transformation=ZeroOneFloat(bounds),
        is_continuous=False,
        is_ordered=False,
        num_values=np.inf,
    )

    check_value_numvalue_conversion(
        p,
        [
            (f, (f - bounds[0]) / (bounds[1] - bounds[0]))
            for f in np.linspace(bounds[0], bounds[1], 128)
        ],
    )
    check_sampling(p)

    ref_ll = -np.log(bounds[1] - bounds[0])

    for v in [-4, 0, 4]:
        assert np.allclose(p.loglikelihood(v), ref_ll, 1e-6)
        assert np.allclose(p.loglikelihood_numerical_value(p.val2num(v)), ref_ll, 1e-6)

    for v in np.linspace(0, 1):
        assert np.allclose(
            p.pdf_numerical_value(v), 1.0 / (bounds[1] - bounds[0]), 1e-6
        )

    assert p != q


@pytest.mark.parametrize(
    "name", ["param-name", "param:name", "param name", "123", "lambda", "def"]
)
def test_raises_on_invalid_name(name):
    with pytest.raises(ValueError, match=name):
        PseudoParameter(
            name,
            prior=Uniform(),
            transformation=ZeroOneFloat(np.array((0, 1))),
            is_continuous=False,
            is_ordered=False,
            num_values=np.inf,
        )
