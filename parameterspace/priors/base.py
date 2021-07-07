# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import abc
import importlib
from typing import Tuple

import numpy as np


class BasePrior(abc.ABC):
    """Base class defining the API of the priors.

    The priors enable the incorporation of domain knowledge into the parameter
    definition by allowing the specification of a PDF/PMF. These are used to sample
    random values and to compute the loglikelihood of a given value.
    """

    def __init__(self, bounds: Tuple):
        """
        Args:
            bounds: Lower and upper bound of the prior.
        """
        self.bounds = np.array(bounds, dtype=float)

    def loglikelihood(self, value):
        """
        Compute the log PDF (up to an additive constant) of a given value.

        Note:
            Values for the priors are always after the transformation!

        Args:
            value: [description]

        Returns:
            [descriptions]
        """
        return np.log(self.pdf(value))

    @abc.abstractmethod
    def pdf(self, value):
        """
        Computes the PDF of a given value.

        Note:
            Values for the priors are always after the transformation!

        Args:
            value: [description]

        Returns:
            [descriptions]
        """

    @abc.abstractmethod
    def sample(self, num_samples: int = 1):
        """
        Draw random samples from the prior.

        Args:
            num_samples: [description]

        Returns:
            [descriptions]
        """

    @staticmethod
    def from_dict(json_dict):
        prior_class = json_dict["class_name"]
        module_str, class_str = prior_class.rsplit(".", 1)
        module = importlib.import_module(module_str)
        model_class = getattr(module, class_str)
        return model_class(*json_dict["init_args"], **json_dict["init_kwargs"])

    def to_dict(self):
        json_dict = {
            "class_name": type(self).__module__ + "." + type(self).__qualname__,
            "init_args": self._init_args,
            "init_kwargs": self._init_kwargs,
        }
        return json_dict

    def __eq__(self, other):
        """Uniform prior doesn't have a state, so equality is just class membership."""
        return isinstance(other, type(self))
