# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import abc
import importlib
from typing import Any, Optional, Tuple

import numpy as np


class BaseTransformation(abc.ABC):
    """Base class defining the API of a transformation."""

    def __init__(self, input_bounds: Optional[Tuple], output_bounds: Tuple):
        """
        Args:
            input_bounds: The numerical range for the input of a transformation. If
                that is not applicable, e.g. for categorical parameters, use `None`.
            output_bounds: The numerical range for the output of a transformation.
                As every transformation yields a numerical representation, these
                bounds have to be always specified in the form
                [lower_bound, upper_bound].
        """
        self.input_bounds = None if input_bounds is None else np.array(input_bounds)
        self.output_bounds = np.array(output_bounds)

    @abc.abstractmethod
    def inverse(self, numerical_value: float) -> Any:
        """Convert the numerical representation back to the true value with the proper
        type.

        Args:
            numerical_value: Transformed/Numerical representation of a value.

        Returns:
            The value corresponding to the given value. Type depends no the kind of
            transformation.
        """

    @abc.abstractmethod
    def __call__(self, value: Any) -> float:
        """Convert a value into the numerical representation.

        Args:
            value: A valid value for this transformation.

        Returns:
            Transformed/Numerical representation of the value.
        """

    def jacobian_factor(self, numerical_value: float) -> float:
        """Factor to correct the likelihood based on the non-linear transformation.

        Args:
            numerical_value: Transformed/Numerical representation of a value.

        Returns:
            Jacobian factor to properly transform the likelihood.
        """
        return 1.0

    @staticmethod
    def from_dict(json_dict: dict):
        """
        [summary]
        """
        transformation_class = json_dict["class_name"]
        module_str, class_str = transformation_class.rsplit(".", 1)
        module = importlib.import_module(module_str)
        model_class = getattr(module, class_str)
        return model_class(*json_dict["init_args"], **json_dict["init_kwargs"])

    def to_dict(self):
        """
        [summary]
        """
        json_dict = {
            "class_name": type(self).__module__ + "." + type(self).__qualname__,
            "init_args": self._init_args,
            "init_kwargs": self._init_kwargs,
        }
        return json_dict
