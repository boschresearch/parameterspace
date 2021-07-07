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
from parameterspace.transformations.log_zero_one import LogZeroOneInteger
from parameterspace.transformations.zero_one import ZeroOneInteger
from parameterspace.utils import store_init_arguments


class IntegerParameter(BaseParameter):
    """Integer parameter that can take every integer value in a given range."""

    @store_init_arguments
    def __init__(
        self,
        name: str,
        bounds: Tuple,
        *,
        prior: Optional[BasePrior] = None,
        transformation: Union[BaseTransformation, str, None] = None,
        inactive_numerical_value: Optional[float] = np.nan
    ):
        """
        Initialize with options for integer parameter.

        Args:
            name: Name of the parameter.
            bounds: Lower and upper bound for the parameter.
            prior: Default prior is the UniformInteger prior.
            transformation: Default transformation is the identity. If a string is
                given, the transform is created automatically for supported transforms.
            inactive_numerical_value: Placeholder value for this parameter in case it
                is not active.
        """
        self.bounds = np.array(bounds, dtype=int)

        if not isinstance(transformation, BaseTransformation):
            if transformation is None:
                transformation = ZeroOneInteger(self.bounds)
            elif transformation == "log":
                transformation = LogZeroOneInteger(self.bounds)

        prior = Uniform() if prior is None else prior

        # TODO: Ask Stefan: inactive_numerical_value is unused/not passed
        super().__init__(
            name,
            prior,
            transformation,
            is_continuous=False,
            is_ordered=True,
            num_values=bounds[1] - bounds[0] + 1,
            inactive_numerical_value=np.nan,
        )

    def __repr__(self):
        """Add bounds to the string representation."""
        string = super().__repr__()
        string += "Bounds: [{}, {}]\n".format(self.bounds[0], self.bounds[1])
        return string

    def check_value(self, value):
        """Check if value is valid."""
        return self.bounds[0] <= value <= self.bounds[1] and value == int(value)

    def num2val(self, numerical_value):
        """Translate the numerical representation into the actual value."""
        return int(super().num2val(numerical_value))
