# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from parameterspace.priors.base import BasePrior
from parameterspace.utils import store_init_arguments


class Uniform(BasePrior):
    """Uninformed prior that puts equal weight on every value."""

    @store_init_arguments
    def __init__(self):
        super().__init__([0, 1])

    def pdf(self, value):
        """Calculate probability density function value.

        Return constant for values inside the bounds, zero if outside, and NaN for NaNs.
        """
        value = np.atleast_1d(value)
        active_idx = np.isfinite(value)
        pdf = np.full(value.shape, np.nan)
        inside = np.logical_and(
            self.bounds[0] <= value[active_idx], value[active_idx] <= self.bounds[1]
        )
        pdf[active_idx] = 1.0 / (self.bounds[1] - self.bounds[0]) * (inside)
        return pdf.squeeze()

    def sample(self, num_samples=None, random_state=np.random):
        return random_state.uniform(
            low=self.bounds[0], high=self.bounds[1], size=num_samples
        )

    def __repr__(self):
        """Minimal information about the Prior."""
        return "Uniform prior in the interval [%f, %f]." % (
            self.bounds[0],
            self.bounds[1],
        )
