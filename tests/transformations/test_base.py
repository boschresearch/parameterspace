# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from parameterspace.transformations.base import BaseTransformation


class DummyTransformation(BaseTransformation):
    def inverse(self, value):
        pass

    def __call__(self, value):
        pass


def test_base_transformation_init():
    t1 = DummyTransformation([1.0, 2.0], [2.0, 3.0])
    np.testing.assert_array_equal(t1.input_bounds, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(t1.output_bounds, np.array([2.0, 3.0]))

    t1 = DummyTransformation(None, [2.0, 3.0])
    assert t1.input_bounds is None
    np.testing.assert_array_equal(t1.output_bounds, np.array([2.0, 3.0]))
