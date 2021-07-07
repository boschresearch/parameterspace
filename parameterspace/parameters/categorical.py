# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple, Union

import numpy as np

from parameterspace.parameters.base import BaseParameter
from parameterspace.priors.categorical import Categorical
from parameterspace.transformations.base import BaseTransformation
from parameterspace.transformations.categorical import Cat2Num
from parameterspace.utils import store_init_arguments


class CategoricalParameter(BaseParameter):
    """Categorical parameter that can take discrete values of any type."""

    @store_init_arguments
    def __init__(
        self,
        name: str,
        values: Tuple[Any],
        prior: Union[list, np.ndarray, None] = None,
        transformation: Optional[BaseTransformation] = None,
        inactive_numerical_value: Optional[float] = np.nan,
    ):
        """
        Initialize with options for categorical parameter.

        Args:
            name: Name of the parameter.
            values: Allowed values for this parameter.
            prior: Probabilities for each value (does not need to be normalized).
            transformation: A transformation that can translate the arbitrary type of
                the values to a numerical value. The only supported one right now
                is [parameterspace.transformations.categorical.Cat2Num][], which is
                also used as default.
            inactive_numerical_value: [description]
        """
        self.values = values
        transformation = Cat2Num(values) if transformation is None else transformation

        if prior is None:
            prior = Categorical([1.0] * len(values))
        elif not isinstance(prior, Categorical):
            prior = Categorical(prior)

        super().__init__(
            name,
            prior,
            transformation,
            is_continuous=False,
            is_ordered=False,
            num_values=len(values),
            inactive_numerical_value=inactive_numerical_value,
        )

    def check_value(self, value):
        """Check if value is valid."""
        return value in self.values
