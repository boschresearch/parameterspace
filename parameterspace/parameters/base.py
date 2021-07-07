# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import abc
import copy
import importlib
from typing import Optional, Union

import numpy as np

from parameterspace.priors.base import BasePrior
from parameterspace.transformations.base import BaseTransformation


class BaseParameter(abc.ABC):
    """Base parameter class defining the API."""

    def __init__(
        self,
        name: str,
        prior: BasePrior,
        transformation: BaseTransformation,
        is_continuous: bool,
        is_ordered: bool,
        num_values: Union[int, float],
        inactive_numerical_value: Optional[float] = np.nan,
    ):
        """
        Initialize with options common to all parameters.

        Args:
            name: Name of the parameter.
            prior: Defines the pdf from which random samples are drawn. Default (`None`)
                correponds to a uniform prior. Note that the prior's bounds must match
                the transformation's output bounds.
            transformation: Defines the tranformation from the values to their numerical
                representaiton. Note that the transformation's output bounds must match
                the prior's bounds.
            is_continuous: Indicates whether this parameter varies continuously, i.e.
                is a `float`.
            is_ordered: Indicates whether this parameter has a natural ordering of it's
                values.
            num_values: Number of possible values. For ordinal, categorical, and integer
                parameters, this equals the number of unique values. For continuous
                parameters, it equals np.inf (eventhough technically there is only a
                finite number of float values).
            inactive_numerical_value: Placeholder value for this parameter in case it
                is not active
        """
        self.name = name
        self._prior = prior
        self._transformation = transformation
        self._inactive_numerical_value = inactive_numerical_value
        self.transformed_bounds = self._transformation.output_bounds

        assert np.allclose(
            self._prior.bounds, self.transformed_bounds, 1e-6
        ), "Missmatch between the prior bounds (%s)" % (
            self._prior.bounds
        ) + "and the transformation's bounds (%s)!" % (
            self.transformed_bounds
        )

        self.is_continuous = is_continuous
        self.is_ordered = is_ordered
        self.num_values = num_values

    def __repr__(self):
        """Basic info about a parameter common to all types."""
        string = "Name: %s\n" % self.name
        string += "Type: %s\n" % self.__class__
        string += "Prior: " + self._prior.__repr__() + "\n"
        string += "is continuous: %s\n" % self.is_continuous
        string += "is ordered: %s\n" % self.is_ordered
        string += "num_values: %s\n" % self.num_values
        return string

    def sample_values(self, num_samples=None, random_state=np.random):
        """Generate randomly sampled values based on the prior."""
        numerical_samples = self.sample_numerical_values(num_samples, random_state)
        if num_samples is None:
            return self._transformation.inverse(numerical_samples)
        return [
            self._transformation.inverse(num_value) for num_value in numerical_samples
        ]

    def sample_numerical_values(self, num_samples=None, random_state=np.random):
        """Generate random values based on the prior, but in the transformed space."""
        return self._prior.sample(num_samples, random_state=random_state)

    def val2num(self, value):
        """Translate a value into its numerical representation (incl. normalization)."""
        if value is None:
            return self._inactive_numerical_value
        return self._transformation(value)

    def num2val(self, numerical_value):
        """Translate the numerical representation into the actual value."""
        return self._transformation.inverse(numerical_value)

    def check_numerical_value(self, numerical_value):
        """Check if the numerical representation of the value is valid."""
        return (
            self.transformed_bounds[0] <= numerical_value <= self.transformed_bounds[1]
        )

    def pdf_numerical_value(self, numerical_value):
        """Compute the PDF based on the prior."""
        return self._prior.pdf(numerical_value) * self._transformation.jacobian_factor(
            numerical_value
        )

    def loglikelihood_numerical_value(self, numerical_value):
        """Compute the loglikelihood based on the prior."""
        return self._prior.loglikelihood(numerical_value) + np.log(
            self._transformation.jacobian_factor(numerical_value)
        )

    def loglikelihood(self, value):
        """Compute the loglikelihood based on the prior."""
        return self.loglikelihood_numerical_value(self._transformation(value))

    def get_numerical_bounds(self):
        """Translate the provided bounds into the numerical representation."""
        return self.transformed_bounds

    @abc.abstractmethod
    def check_value(self, value):
        """Checks if value is valid."""

    @staticmethod
    def from_dict(json_dict):
        parameter_class = json_dict["class_name"]
        module_str, class_str = parameter_class.rsplit(".", 1)
        module = importlib.import_module(module_str)
        parameter_class = getattr(module, class_str)

        transformation = BaseTransformation.from_dict(json_dict["transformation"])
        prior = BasePrior.from_dict(json_dict["prior"])

        return parameter_class(
            *json_dict["init_args"],
            **json_dict["init_kwargs"],
            transformation=transformation,
            prior=prior
        )

    def to_dict(self):
        json_dict = {
            "class_name": type(self).__module__ + "." + type(self).__qualname__,
            "init_args": self._init_args,
            "init_kwargs": copy.deepcopy(self._init_kwargs),
            "transformation": self._transformation.to_dict(),
            "prior": self._prior.to_dict(),
        }
        for key in ["transformation", "prior"]:
            if key in json_dict["init_kwargs"]:
                del json_dict["init_kwargs"][key]

        return json_dict

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.name != other.name:
            return False
        return (self._prior == other._prior) and (
            self._transformation == other._transformation
        )
