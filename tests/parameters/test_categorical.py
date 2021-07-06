# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from parameterspace.parameters.base import BaseParameter
from parameterspace.parameters.categorical import CategoricalParameter

from .util import check_sampling, check_value_numvalue_conversion, check_values


def test_categorical_parameter():
    values = [4, 3, 2, 1, 0]

    p = CategoricalParameter("foo", values, prior=[1] * len(values))

    assert p.is_continuous == False
    assert p.is_ordered == False

    check_values(p, [-1, 2, "foo", 4], [False, True, False, True])
    check_value_numvalue_conversion(
        p,
        [
            (values.index(i), (i + 0.5) / len(values))
            for i in np.random.randint(len(values), size=128)
        ],
    )

    # test string representation
    str(p)


def test_categorical_parameter_mixed_types():
    values = ["foo", "bar", "baz", 42]

    p = CategoricalParameter("foo", values)

    assert p.is_continuous == False
    assert p.is_ordered == False

    check_values(p, ["foo", "bar", "foobar", 42], [True, True, False, True])

    reference = [
        (values[i], (i + 0.5) / len(values))
        for i in np.random.randint(len(values), size=128)
    ]
    check_value_numvalue_conversion(p, reference, num_values=False)


def test_categorical_parameter_to_from_dict():
    values = ["foo", "bar", "baz", 42]

    p1 = CategoricalParameter("foo", values)

    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BaseParameter.from_dict(json_dict)
    assert p1 == p2

    check_values(p2, ["foo", "bar", "foobar", 42], [True, True, False, True])
    reference = [
        (values[i], (i + 0.5) / len(values))
        for i in np.random.randint(len(values), size=128)
    ]
    check_value_numvalue_conversion(p2, reference, num_values=False)


def test_categorical_parameter_equal():
    values = ["foo", "bar", "baz", 42]

    p1 = CategoricalParameter("foo", values)
    p2 = CategoricalParameter("foo", values + ["whatever"])

    assert p1 != p2


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
