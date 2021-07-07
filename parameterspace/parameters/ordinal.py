# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import numpy as np

from parameterspace.parameters.base import BaseParameter
from parameterspace.priors.uniform import Uniform
from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.categorical import Cat2Num
from parameterspace.utils import store_init_arguments


class OrdinalParameter(BaseParameter):
    """Ordinal parameter that can take discrete values of any type."""

    @store_init_arguments
    def __init__(
        self,
        name: str,
        values: list,
        *,
        prior: Union[list, np.ndarray, None] = None,
        transformation: Optional[BaseTransformation] = None,
        inactive_numerical_value: Optional[float] = np.nan
    ):
        """
        Initialize with options for ordinal parameter.

        Args:
            name: Name of the parameter.
            values: Allowed values in ascending order for this parameter.
            prior: List of (unnormalized) probablities for each value.
                Default puts equal probablity for each value.
            transformation: The only supported transformation right now
                is [parameterspace.transformations.categorical.Cat2Num][],
                which is also the default.
            inactive_numerical_value: Placeholder value for this parameter
                in case it is not active. Default is NaN.
        """
        self.values = values
        transformation = Cat2Num(values) if transformation is None else transformation
        prior = Uniform() if prior is None else prior
        super().__init__(
            name,
            prior,
            transformation,
            is_continuous=False,
            is_ordered=True,
            num_values=len(values),
            inactive_numerical_value=inactive_numerical_value,
        )

    def check_value(self, value):
        """Check if value is valid."""
        return value in self.values
