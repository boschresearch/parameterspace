# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple

import numpy as np

from parameterspace.transformations.base import BaseTransformation
from parameterspace.utils import store_init_arguments


class LogZeroOneFloat(BaseTransformation):
    """Maps bounded interval to [0,1] via a logarithmic transformation.

    This means that all the priors used with a parameter effectively model the
    exponent of the actual quantity.

    This class should be used for ContinuousParameters.
    """

    @store_init_arguments
    def __init__(self, bounds: Optional[Tuple]):
        """
        [summary]

        Args:
            bounds: Bounds must be positive for log-transformation.
        """
        assert min(bounds) > 0, "bounds must be positive for log-transformation"
        super().__init__(bounds, (0, 1))
        self.log_bounds = np.log(self.input_bounds)
        self.log_interval_size = self.log_bounds[1] - self.log_bounds[0]

    def inverse(self, numerical_value: float) -> float:
        return float(
            np.clip(
                np.exp(self.log_bounds[0] + numerical_value * (self.log_interval_size)),
                self.input_bounds[0],
                self.input_bounds[1],
            )
        )

    def __call__(self, value: Any) -> float:
        return float((np.log(value) - self.log_bounds[0]) / self.log_interval_size)

    def jacobian_factor(self, numerical_value: float) -> float:
        return 1.0 / (self.log_interval_size * self.inverse(numerical_value))

    def __eq__(self, other):
        return np.allclose(self.input_bounds, other.input_bounds)


class LogZeroOneInteger(BaseTransformation):
    """Maps a bounded interval of integers to [0, 1] via a logarithmic transformation.

    This means that all the priors used with a parameter effectively model the
    exponent of the actual quantity.

    This class should be used for IntegerParameters.
    """

    @store_init_arguments
    def __init__(self, bounds: Optional[Tuple]):
        """
        [summary]

        Args:
            bounds: Bounds must be positive for log-transformation.
        """
        assert min(bounds) > 0, "bounds must be positive for log-transformation"
        super().__init__(bounds, (0, 1))
        self.log_bounds = np.log(self.input_bounds + np.array([-0.5, 0.5]))
        self.log_interval_size = self.log_bounds[1] - self.log_bounds[0]

    def inverse(self, numerical_value: float) -> int:
        """
        [summary]
        """
        return int(
            np.around(
                np.exp(self.log_bounds[0] + numerical_value * (self.log_interval_size))
            )
        )

    def __call__(self, value: Any) -> float:
        return float((np.log(value) - self.log_bounds[0]) / self.log_interval_size)

    def jacobian_factor(self, numerical_value: float) -> float:
        return 1.0 / (self.log_interval_size * self.inverse(numerical_value))

    def __eq__(self, other):
        return np.allclose(self.input_bounds, other.input_bounds)
