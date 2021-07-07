# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import Any

import numpy as np

from parameterspace.transformations.base import BaseTransformation
from parameterspace.utils import store_init_arguments


class Cat2Num(BaseTransformation):
    """Translates any values into discrete, equidistant values between 0 and 1."""

    @store_init_arguments
    def __init__(self, values: list):
        """
        Args:
            values: List of all possible values of (almost) arbitrary type.
        """
        super().__init__(None, (0, 1))
        self.values = copy.deepcopy(values)

    def inverse(self, numerical_value: float) -> Any:
        return self.values[
            int(
                np.clip(
                    np.around(numerical_value * len(self.values) - 0.5),
                    0,
                    len(self.values) - 1,
                )
            )
        ]

    def __call__(self, value: Any) -> float:
        idx = self.values.index(value)
        return (0.5 + idx) / len(self.values)

    def __eq__(self, other):
        if len(self.values) != len(other.values):
            return False
        for e1, e2 in zip(self.values, other.values):
            # need to handle some special cases that can occur with categorical values
            if (e1 is None) and (e2 is None):
                continue
            if e1 == e2:
                continue
            # == will not yield true, if both values are NAN
            try:
                if math.isnan(e1) and math.isnan(e2):
                    continue
            except TypeError:
                pass

            return False

        return True
