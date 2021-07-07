# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def check_transform_and_inverse(t, value_pairs):
    for v, tv in value_pairs:
        assert np.allclose(t(v), tv, 1e-6)
        assert np.allclose(t.inverse(tv), v, 1e-6)


def check_bounds(t, expected_in, expected_out):
    assert np.allclose(t.input_bounds, expected_in, 1e-6)
    assert np.allclose(t.output_bounds, expected_out, 1e-6)
