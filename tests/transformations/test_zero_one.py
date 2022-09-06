# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.zero_one import ZeroOneFloat, ZeroOneInteger

from .util import check_bounds, check_transform_and_inverse


def test_zero_one_integer_transformation():
    bounds = [-4, 8]
    t = ZeroOneInteger(bounds)
    reference_values = [
        (i + bounds[0], (i + 0.5) / (bounds[1] - bounds[0] + 1)) for i in range(13)
    ]

    check_transform_and_inverse(t, reference_values)
    check_bounds(t, bounds, [0, 1])


def test_zero_one_integer_to_from_dict():
    bounds = [-4, 8]
    t1 = ZeroOneInteger(bounds)
    reference_values = [
        (i + bounds[0], (i + 0.5) / (bounds[1] - bounds[0] + 1)) for i in range(13)
    ]

    json_dict = t1.to_dict()
    json.dumps(json_dict)
    t2 = BaseTransformation.from_dict(json_dict)

    check_transform_and_inverse(t2, reference_values)
    check_bounds(t2, bounds, [0, 1])
    assert t1 == t2


def test_zero_one_float_transformation():
    bounds = [-4, 8]
    t = ZeroOneFloat(bounds)
    reference_values = [
        (f, (f - bounds[0]) / (bounds[1] - bounds[0]))
        for f in np.linspace(bounds[0], bounds[1], 128)
    ]

    check_transform_and_inverse(t, reference_values)
    check_bounds(t, bounds, [0, 1])


def test_zero_one_float_to_from_dict():
    bounds = [-4, 8]
    t1 = ZeroOneFloat(bounds)
    reference_values = [
        (f, (f - bounds[0]) / (bounds[1] - bounds[0]))
        for f in np.linspace(bounds[0], bounds[1], 128)
    ]

    json_dict = t1.to_dict()
    json.dumps(json_dict)
    t2 = BaseTransformation.from_dict(json_dict)

    check_transform_and_inverse(t2, reference_values)
    check_bounds(t2, bounds, [0, 1])
    assert t1 == t2
