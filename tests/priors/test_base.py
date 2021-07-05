# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np

from parameterspace.priors import Beta, Categorical, TruncatedNormal, Uniform


def test_equal():
    tn = TruncatedNormal(0.5, 2)
    u = Uniform()
    c = Categorical([0.5, 0.5])
    b = Beta(1, 1)

    for d1, d2 in itertools.combinations([tn, u, c, b], 2):
        assert d1 != d2
