# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import numpy as np

from parameterspace.priors.base import BasePrior
from parameterspace.utils import store_init_arguments


class Categorical(BasePrior):
    """Categorical prior with separate probabilities for each value."""

    @store_init_arguments
    def __init__(self, prior_values: Union[list, np.ndarray]):
        """
        Args:
            prior_values: Array containing the probabilities for each value.
        """
        super().__init__((0, 1))
        self.probabilities = np.array(prior_values)
        if not np.all(self.probabilities >= 0):
            raise ValueError("Probabilities must be non-negative!")

        self.probabilities = self.probabilities / np.sum(self.probabilities)
        self.numerical_values = (
            np.arange(len(self.probabilities), dtype=float) + 0.5
        ) / len(self.probabilities)

    def loglikelihood(self, value):
        return np.log(self.pdf(value))

    def pdf(self, value):
        """Probability given the numerical representation(s).

        The numerical representation is converted to integers and the corresponding
        prior_values are returned.
        If the value exceeds the given range of values, a ValueError is raised.

        Only finite values are converted, and the respective probability is computed.
        NaN and INF will result in a NaN. For example:
            value = [nan, inf, 0]
        will result in
            [NaN, NaN, p(0)].

        """
        value = np.atleast_1d(value)
        idx = np.isfinite(value)

        integer_value = np.around(len(self.probabilities) * value[idx] - 0.5).astype(
            int
        )
        pdf = np.full(value.shape, np.nan, dtype=float)

        try:
            pdf[idx] = self.probabilities[integer_value]
        except IndexError:
            raise ValueError(
                "Unknown value in the numerical representation for a "
                + "categorical parameter encountered!"
            )
        return pdf.squeeze()

    def sample(self, num_samples=None, random_state=np.random):
        return random_state.choice(
            self.numerical_values, size=num_samples, replace=True, p=self.probabilities
        )

    def __repr__(self):
        return "Categorical prior for %i values with p = %s" % (
            len(self.probabilities),
            self.probabilities,
        )

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and (len(self.probabilities) == len(other.probabilities))
            and np.allclose(self.probabilities, other.probabilities)
        )
