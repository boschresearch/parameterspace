# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple

import numpy as np

from parameterspace.transformations.base import BaseTransformation
from parameterspace.utils import store_init_arguments


class ZeroOneFloat(BaseTransformation):
    """Maps a bounded interval to [0, 1] via a linear transformation."""

    @store_init_arguments
    def __init__(self, bounds: Optional[Tuple]):
        super().__init__(bounds, (0, 1))
        self.interval_size = bounds[1] - bounds[0]

    def inverse(self, numerical_value: float) -> float:
        return float(
            np.clip(
                self.input_bounds[0] + numerical_value * (self.interval_size),
                self.input_bounds[0],
                self.input_bounds[1],
            )
        )

    def __call__(self, value: Any) -> float:
        return float((value - self.input_bounds[0]) / self.interval_size)

    def __eq__(self, other):
        return np.allclose(self.input_bounds, other.input_bounds)

    def jacobian_factor(self, numerical_value: float) -> float:
        return 1.0 / self.interval_size


class ZeroOneInteger(BaseTransformation):
    """Maps a bounded interval of integers to [0, 1] via a linear transformation."""

    @store_init_arguments
    def __init__(self, bounds: Optional[Tuple]):
        super().__init__(bounds, (0, 1))
        self.interval_size = bounds[1] - bounds[0] + 1

    def inverse(self, numerical_value: float) -> int:
        return int(
            np.clip(
                np.around(
                    self.input_bounds[0] - 0.5 + numerical_value * (self.interval_size)
                ),
                self.input_bounds[0],
                self.input_bounds[1],
            )
        )

    def __call__(self, value: Any) -> float:
        return float((value - self.input_bounds[0] + 0.5) / self.interval_size)

    def __eq__(self, other):
        return np.allclose(self.input_bounds, other.input_bounds)

    def jacobian_factor(self, numerical_value: float) -> float:
        return 1.0 / self.interval_size
