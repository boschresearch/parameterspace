# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json
import math

import numpy as np
import pytest

from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.categorical import Cat2Num


def test_cat2num_transformation():
    values = ["foo", "bar", 1, 2, 3]
    t = Cat2Num(values)
    reference_values = [
        (values[i], (i + 0.5) / len(values)) for i in np.random.choice(len(values), 128)
    ]
    # check_* functions in util don't work, b/c the types are not necessarily numerical
    ref_original = [v[0] for v in reference_values]
    ref_transformed = [v[1] for v in reference_values]

    all_transformed = [t(v) for v in ref_original]
    assert np.allclose(all_transformed, ref_transformed)

    all_inverted = [t.inverse(v) for v in ref_transformed]

    for original, inv_transformed in zip(ref_original, all_inverted):
        assert original == inv_transformed

    for v, tv in reference_values:
        assert t(v) == tv
        assert t.inverse(tv) == v

    assert t.inverse(-1e-4) == values[0]
    assert t.inverse(1 + 1e-4) == values[-1]

    assert np.allclose(t.output_bounds, [0, 1], 1e-6)
    assert t.input_bounds is None


def test_cat2num_to_from_dict():
    values = ["foo", "bar", 1, 2, 3.0, None, float("NaN")]
    t1 = Cat2Num(values)
    json_dict = t1.to_dict()
    json.dumps(json_dict)
    t2 = BaseTransformation.from_dict(json_dict)

    assert t1 == t2

    # for good measure, let's make sure that both transform values to back and forth in the same way
    reference_values = [
        (values[i], (i + 0.5) / len(values)) for i in np.random.choice(len(values), 128)
    ]
    # check_* functions in util don't work, b/c the types are not necessarily numerical
    ref_original = [v[0] for v in reference_values]
    ref_transformed = [v[1] for v in reference_values]

    all_inverted1 = [t1.inverse(v) for v in ref_transformed]
    all_inverted2 = [t2.inverse(v) for v in ref_transformed]

    for v, tv in reference_values:
        assert t1(v) == tv
        assert t1.inverse(tv) == v or (math.isnan(t1.inverse(tv)) and math.isnan(v))
        assert t2(v) == tv
        assert t2.inverse(tv) == v or (math.isnan(t2.inverse(tv)) and math.isnan(v))


def test_cat2num_equal():
    values = ["foo", "bar", 1, 2, 3.0, None, float("NaN")]

    t1 = Cat2Num(values)
    t2 = Cat2Num(values[:-1])

    values[0] = "baz"

    t3 = Cat2Num(values)

    assert t1 != t2
    assert t1 != t3


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
