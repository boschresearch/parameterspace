# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple, Union

import numpy as np

from parameterspace.parameters.base import BaseParameter
from parameterspace.priors.base import BasePrior
from parameterspace.priors.uniform import Uniform
from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.log_zero_one import LogZeroOneFloat
from parameterspace.transformations.zero_one import ZeroOneFloat
from parameterspace.utils import store_init_arguments


class ContinuousParameter(BaseParameter):
    """Continuous parameter represented by a float in a given range."""

    @store_init_arguments
    def __init__(
        self,
        name: str,
        bounds: Tuple[float, float],
        *,
        prior: Optional[BasePrior] = None,
        transformation: Union[BaseTransformation, str, None] = None,
        inactive_numerical_value: Optional[float] = np.nan,
    ):
        """
        Initialize with options for continuous parameter.

        Args:
            name: Name of the parameter.
            bounds: Lower and upper bound for the parameter.
            prior: Default prior is the Uniform prior which will fail if at least one
                of the bounds is infinite!
            transformation: Default transformation is the affine-linear transformation
                of the bounds to `[0,1]`. If a string is given, the transform is created
                automatically for supported transforms.
            inactive_numerical_value:  Placeholder value for this parameter in case
                it is not active.
        """
        if not np.isfinite(bounds).all():
            raise ValueError(f"Bounds need to be finite, but are {bounds}")

        self.bounds = np.array(bounds)

        if not isinstance(transformation, BaseTransformation):
            if transformation is None:
                transformation = ZeroOneFloat(self.bounds)
            elif transformation == "log":
                transformation = LogZeroOneFloat(self.bounds)

        prior = Uniform() if prior is None else prior
        super().__init__(
            name,
            prior,
            transformation,
            is_continuous=True,
            is_ordered=True,
            num_values=np.inf,
            inactive_numerical_value=inactive_numerical_value,
        )

    def __repr__(self):
        """Add bounds to the string representation."""
        string = super().__repr__()
        string += f"Bounds: ({self.bounds[0]}, {self.bounds[1]})\n"
        return string

    def check_value(self, value):
        """Check if value is valid."""
        return self.bounds[0] <= value <= self.bounds[1]
