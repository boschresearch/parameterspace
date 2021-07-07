# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as sps

from parameterspace.priors.base import BasePrior
from parameterspace.utils import store_init_arguments


class Beta(BasePrior):
    """Beta prior for variables in the interval [0,1]."""

    @store_init_arguments
    def __init__(self, a: float, b: float):
        """
        Args:
            a: Positive parameter of the Beta distribution.
            b: Positive parameter of the Beta distribution.
        """
        super().__init__((0, 1))
        self.a, self.b = a, b
        self.sps_beta_dist = sps.beta(a, b)

    def pdf(self, value):
        return self.sps_beta_dist.pdf(value)

    def loglikelihood(self, value):
        return self.sps_beta_dist.logpdf(value)

    def sample(self, num_samples=None, random_state=np.random):
        return self.sps_beta_dist.rvs(size=num_samples, random_state=random_state)

    def __repr__(self):
        return "Beta distribution with parameteres a=%f, b=%f" % (self.a, self.b)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            [self.a, self.b], [other.a, other.b]
        )
