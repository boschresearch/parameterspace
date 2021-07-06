# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as sps

from parameterspace.priors.base import BasePrior


class Normal(BasePrior):
    """Stardand Gaussian prior."""

    def __init__(self, mean=0, std=1):
        super().__init__(np.array([-np.inf, np.inf]))
        self.sps_normal_dist = sps.norm(loc=mean, scale=std)

    def loglikelihood(self, value):
        return self.sps_normal_dist.logpdf(value)

    def pdf(self, value):
        return self.sps_normal_dist.pdf(value)

    def sample(self, num_samples=None, random_state=np.random):
        return self.sps_normal_dist.rvs(size=num_samples, random_state=random_state)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            [self.mean, self.std], [other.mean, other.std]
        )
