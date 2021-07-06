# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.random as npr
import scipy.stats as sps

from parameterspace.priors.base import BasePrior
from parameterspace.utils import store_init_arguments


class TruncatedNormal(BasePrior):
    """Truncated normal prior for bounded parameters."""

    @store_init_arguments
    def __init__(self, mean: float = 0.5, std: float = 1):
        """
        Args:
            mean: Mean of the distribution.
            std: Standard deviation of the distribution
        """
        super().__init__((0, 1))
        a, b = (self.bounds - mean) / std
        self.mean = mean
        self.std = std
        self.sps_dist = sps.truncnorm(a, b, loc=mean, scale=std)

    def pdf(self, value):
        return self.sps_dist.pdf(value)

    def loglikelihood(self, value):
        return self.sps_dist.logpdf(value)

    def sample(self, num_samples=None, random_state=npr):
        return self.sps_dist.rvs(size=num_samples, random_state=random_state)

    def __repr__(self):
        return (
            "Truncated normal (Interval [%f, %f]) with parameteres mean=%f, std=%f"
            % (self.bounds[0], self.bounds[1], self.mean, self.std)
        )

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            [self.mean, self.std], [other.mean, other.std]
        )
