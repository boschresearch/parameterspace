# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.log_zero_one import (
    LogZeroOneFloat,
    LogZeroOneInteger,
)

from .util import check_bounds, check_transform_and_inverse


def test_log_zero_one_integer_transformation():
    bounds = (1, 1024)
    t = LogZeroOneInteger(bounds)
    ref_values = [2**i for i in range(11)]

    ref_values_transformed = (np.log(ref_values) - np.log(bounds[0] - 0.5)) / (
        np.log(bounds[1] + 0.5) - np.log(bounds[0] - 0.5)
    )

    check_transform_and_inverse(t, zip(ref_values, ref_values_transformed))
    check_bounds(t, bounds, [0, 1])


def test_log_zero_one_integer_to_from_dict():
    bounds = (1, 1024)
    t1 = LogZeroOneInteger(bounds)
    ref_values = [2**i for i in range(11)]
    ref_values_transformed = (np.log(ref_values) - np.log(bounds[0] - 0.5)) / (
        np.log(bounds[1] + 0.5) - np.log(bounds[0] - 0.5)
    )

    json_dict = t1.to_dict()
    json.dumps(json_dict)
    t2 = BaseTransformation.from_dict(json_dict)

    check_transform_and_inverse(t2, zip(ref_values, ref_values_transformed))
    check_bounds(t2, bounds, [0, 1])
    assert t1 == t2


def test_log_zero_one_float_transformation():
    bounds = np.array((1e-2, 1e2))
    t = LogZeroOneFloat(bounds)
    ref_values_transformed = np.linspace(0, 1, 32)
    ref_values = np.power(10, (4 * ref_values_transformed) - 2)

    check_transform_and_inverse(t, zip(ref_values, ref_values_transformed))
    check_bounds(t, bounds, [0, 1])


def test_log_zero_one_float_to_from_dict():
    bounds = np.array((1e-2, 1e2))
    t1 = LogZeroOneFloat(bounds=bounds)
    ref_values_transformed = np.linspace(0, 1, 32)
    ref_values = np.power(10, (4 * ref_values_transformed) - 2)

    json_dict = t1.to_dict()
    json.dumps(json_dict)
    t2 = BaseTransformation.from_dict(json_dict)

    check_transform_and_inverse(t2, zip(ref_values, ref_values_transformed))
    check_bounds(t2, bounds, [0, 1])
    assert t1 == t2
