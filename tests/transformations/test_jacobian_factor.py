# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.optimize import approx_fprime, check_grad

from parameterspace.transformations.log_zero_one import (
    LogZeroOneFloat,
    LogZeroOneInteger,
)
from parameterspace.transformations.zero_one import ZeroOneFloat, ZeroOneInteger


# Note: The integer counterparts will fail, because the inverse is
@pytest.mark.parametrize(
    "transformation_cls,bounds,gradient_bounds",
    [
        (LogZeroOneFloat, [1e-3, 1e3], [0, 1]),
        (LogZeroOneInteger, [1, 1024 ** 3], [0.5, 1]),
        (ZeroOneFloat, [0, 1e3], [0, 1]),
        (ZeroOneInteger, [1, 1024 ** 3], [0, 1]),
    ],
)
def test_jacobian_factor(transformation_cls, bounds, gradient_bounds):
    t = transformation_cls(bounds=bounds)

    # note: numerical approximation will fail at x=0 and x=1 because outside the inverse is constant!
    x_test = np.linspace(*gradient_bounds, 66)[1:-1]

    def tmp_inverse(x):
        """Little helper, because approx_fprime needs an array input, but t.inverse only accepts single values!"""
        return t.inverse(x.squeeze())

    for x in x_test:
        true = t.jacobian_factor(x)
        approx = 1 / approx_fprime(np.array([x]), tmp_inverse, 1e-4)
        assert np.allclose([true], [approx], rtol=1e-3)


if __name__ == "__main__":
    pytest.main(("--pdb %s" % __file__).split())
