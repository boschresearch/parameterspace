import abc

import numpy as np


class SearchSpace(abc.ABC):
    """Abstract base class for different search space implementations."""

    def __init__(self, seed: int = None) -> None:
        """Initialize a search space with an optional seed."""
        self.seed(seed)

    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of parameters in the space."""

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        """Reinitialize the random number generator with a new seed."""

    @abc.abstractmethod
    def copy(self):
        """Get a copy that behaves exactly like the original `SearchSpace`.
        Call `seed()` on the copy to get independent samples.
        """

    @abc.abstractmethod
    def sample(self) -> dict:
        """Provide a dictionary with one key corresponding to each parameter name and
        its value representing a sample for that parameter.
        """

    @abc.abstractmethod
    def to_numerical(self, configuration: dict) -> np.ndarray:
        """Given a configuration from this space, create a numerical vector
        representation.
        The transformed representation needs to be between 0 and 1 (uniform), including
        integers, ordinal and categoricals.
        Inactive parameters have to be represented with `np.nan`
        """

    @abc.abstractmethod
    def from_numerical(self, vector: np.ndarray) -> dict:
        """Convert a np.float64 type vector numerical representation of a configuration
        from this space to a dictionary representation."""
