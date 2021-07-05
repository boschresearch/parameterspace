# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from parameterspace.parameters.ordinal import OrdinalParameter
from parameterspace.parameters.base import BaseParameter

from .util import check_value_numvalue_conversion, check_sampling, check_values


def test_ordinal_parameter():
    values = ["freezing cold", "cold", "warm", "hot"]
    p1 = OrdinalParameter("foo", values)

    assert p1.is_continuous == False
    assert p1.is_ordered == True

    check_values(p1, ["freezing cold", "cool", "warm", np.pi, ["hot"]], [True, False, True, False, False])

    reference = [(values[i], (i + 0.5) / len(values)) for i in [0, 1, 2, 3]]
    check_value_numvalue_conversion(p1, reference, num_values=False)

    json_dict = p1.to_dict()
    json.dumps(json_dict)
    p2 = BaseParameter.from_dict(json_dict)
    assert p1 == p2
    check_value_numvalue_conversion(p2, reference, num_values=False)
