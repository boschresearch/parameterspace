# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

# For type hinting custom class (ParameterSpace):
from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import parameterspace as ps
from parameterspace.base import SearchSpace
from parameterspace.condition import Condition
from parameterspace.parameters.base import BaseParameter


class ParameterSpace(SearchSpace):
    """Class representing a parameter space that allows to sampling, converting and
    checking configurations.
    """

    def __init__(self, seed: int = None):
        super().__init__(seed=seed)

        self._parameters: dict = {}
        self._constants: dict = {}

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __getitem__(self, name):
        return self._parameters[name]

    def __repr__(self):
        string = "ParameterSpace with the following parameteres and conditions\n"
        for p in self._parameters.values():
            string += str(p["parameter"])
            string += str(p["condition"]) + "\n"
        return string

    def __eq__(self, other):
        self_dict = self.to_dict()
        self_dict.pop("bit_generator_state")

        other_dict = other.to_dict()
        other_dict.pop("bit_generator_state")

        return self_dict == other_dict

    def seed(self, seed: Union[int, dict] = None) -> None:
        """Reinitialize the random number generator.

        Args:
            seed: Either an integer seed or a numpy bit generator state dictionary.
                Defaults to None.
        """
        if isinstance(seed, dict):
            self._rng = np.random.default_rng()
            self._rng.bit_generator.state = seed
        else:
            self._rng = np.random.default_rng(seed)

    def copy(self) -> ParameterSpace:
        return ParameterSpace.from_dict(self.to_dict())

    def add(
        self,
        parameter: Union[BaseParameter, ParameterSpace],
        condition: Callable = None,
    ):
        """Add a parameter that is only active if condition returns true when called.

        Args:
            parameter: A parameter or a subspace to be added to this space.\n
                Every parameter needs to have a unique name! A `ValueError` is raised if
                a parameter with the same name already exists in the `ParameterSpace`.
            condition: Lambda function that returns whether this parameter is active
                based on values of other parameters in the `ParameterSpace`.\n
                The signature is read using introspection to extract on which parameters
                the condition operates on.
        """
        if not isinstance(condition, Condition):
            tmp_condition = Condition(condition)
            condition = Condition()
            for n in tmp_condition.all_varnames:
                condition.merge(self._parameters[n]["condition"])

            condition.merge(tmp_condition)

        if isinstance(parameter, ParameterSpace):
            for pn in parameter.get_parameter_names():
                p = parameter[pn]
                tmp_condition = copy.deepcopy(condition).merge(p["condition"])
                self.add(p["parameter"], tmp_condition)
        else:
            if parameter.name in self._parameters:
                raise ValueError("Parameter %s already exists!" % parameter.name)
            self._parameters[parameter.name] = {
                "parameter": parameter,
                "condition": condition,
            }

    def to_dict(self) -> dict:
        """
        Transform current `ParameterSpace` into a dictionary representation.

        Returns:
            Contains "constants" and "parameters" (with "condition").
        """

        parameters = {}
        for p in self._parameters.values():
            parameters[p["parameter"].name] = {
                "parameter": p["parameter"].to_dict(),
                "condition": p["condition"].to_dict(),
            }
        return {
            "bit_generator_state": self._rng.bit_generator.state,
            "parameters": parameters,
            "constants": copy.deepcopy(self._constants),
        }

    @staticmethod
    def from_dict(dict_representation: dict) -> ParameterSpace:
        """
        Create a new `ParameterSpace` instance with the state passed as
        dictionary.

        Args:
            dict_representation: A dictionary representation of a
                `ParameterSpace`.

        Returns:
            New class instance
        """
        return_ps = ParameterSpace()
        return_ps.seed(dict_representation.get("bit_generator_state"))

        for parameter_dict in dict_representation["parameters"].values():
            parameter = BaseParameter.from_dict(parameter_dict["parameter"])
            condition = Condition.from_dict(parameter_dict["condition"])
            return_ps._parameters[parameter.name] = {
                "parameter": parameter,
                "condition": condition,
            }
        return_ps._constants = copy.deepcopy(dict_representation["constants"])

        return return_ps

    def fix(self, **kwargs: Any):
        """Remove all parameters from the parameters list and add treat them as
        constants.

        The main use-case for this method is to narrow down a generic
        `ParameterSpace` by fixing certain variables. This could be because of
        special constraints, or to focus any optimization on a few parameters.

        Note:
            This method will change the ParameterSpace object irreversibly.

        Args:
            **kwargs: All arguments to this method have to be keyword arguments
                with valid parameter names and valid values.\n
                The method will change the current `ParameterSpace` such that
                thenumerical representation does not contain any fixed
                parameters (or always inactive ones).\n
                The dictionary representation will still contain the fixed
                parameters with their respective value.

        Raises:
            ValueError: In case of not existing parameters or invalid values.

        """

        for name, value in kwargs.items():
            if name not in self._parameters:
                raise ValueError(
                    f"Parameter '{name}' is not part of this ParameterSpace!"
                )

            if not self._parameters[name]["parameter"].check_value(value):
                raise ValueError(f"Invalid value `{value}` for parameter '{name}'!")

        # remove any constants here that might not be active based on some parameter values,
        # but are still specified here.
        actual_constants = copy.copy(kwargs)
        for n in kwargs.keys():
            p = self._parameters[n]
            if p["condition"].all_varnames <= set(kwargs) and not p["condition"](
                kwargs
            ):
                actual_constants.pop(n, None)
        self._constants.update(**actual_constants)

        for p in list(self._parameters.keys()):

            if p in self._constants:
                del self._parameters[p]
                continue

            try:
                if self[p]["condition"](self._constants):
                    self[p]["condition"] = Condition(None)
                else:
                    del self._parameters[p]
            except ValueError:
                pass

    def remove(self, name: str):
        """Remove a parameter by name from this `ParameterSpace` instance.

        Note:
            When removing a parameter from a `ParameterSpace` that has been
            added to another `ParameterSpace`, the including `ParameterSpace`
            is not affected.

        Args:
            name: Name of the parameter.

        Raises:
            RuntimeError: In case other parameters condition on the parameter
                that should be removed.
            KeyError: In case parameter doesn't exist.
        """
        conditioning_parameters = []
        for p_name, p in self._parameters.items():
            if name in p["condition"].all_varnames:
                conditioning_parameters.append(p_name)
        if conditioning_parameters:
            raise RuntimeError(
                f"Unable to remove parameter '{name}' "
                + f"because the parameters {conditioning_parameters} condition on it. "
                + f"Please remove the other parameters before removing '{name}'."
            )

        try:
            self._parameters.pop(name)
        except KeyError as e:
            raise KeyError(
                f"Parameter '{name}' is not part of the ParameterSpace."
            ) from e

    def get_parameter_names(self) -> List[str]:
        """Get names of all parameters already in the current `ParameterSpace`.

        Returns:
            Parameter names.
        """
        return list(self._parameters.keys())

    def sample(
        self, partial_configuration: dict = None, rng: np.random.Generator = None
    ) -> dict:
        """Sample a random configuration based on the priors.

        Args:
            partial_configuration: Partial assignment of certain variables. If
                not `None`, only the remaining variables are set randomly. All
                conditions are honored.\n
                Right now, there is no check if the partitial configuration
                violates any conditions.
            rng: A Numpy random number generator used to generate the
                sample. Overrides the internal random number generator that is
                set on initialization of the space.

        Raises:
            ValueError: If a passed value is not valid for the corresponding
                parameter.

        Returns:
            Random parameter configuration.
        """
        if rng is None:
            rng = self._rng

        config = {} if partial_configuration is None else partial_configuration
        for n, v in config.items():
            if not self._parameters[n]["parameter"].check_value(v):
                raise ValueError("%s = %s is not valid for this space" % (n, v))

        config.update(self._constants)

        for n, p in self._parameters.items():
            if n in config:
                continue
            if p["condition"].empty() or p["condition"](config):
                config[n] = p["parameter"].sample_values(random_state=rng)

        return config

    def log_likelihood(self, configuration: dict) -> float:
        """Compute log-likelihood of a configuration under the given prior.

        Args:
            configuration: A parameter configuration

        Returns:
            Log-likelihood value or `NaN` if configuration is invalid.
        """
        if not self.check_validity(configuration):
            return np.nan
        self.remove_inactive(configuration)
        return self.log_likelihood_numerical(self.to_numerical(configuration))

    def log_likelihood_numerical(self, vector_configuration: np.ndarray) -> float:
        """Compute log-likelihood for the numerical representation of a
        configuration under the given prior.

        Note:
            This method assumes that the vector representation is valid! There
            are no sanity checks at this point. Inactive parameters must have
            `NaN` values.

        Args:
            vector_configuration: Numerical vector representation of a
                configuration.

        Returns:
            Log-likelihood value
        """
        ll = 0.0
        for i, p in enumerate(self._parameters.values()):
            if np.isfinite(vector_configuration[i]):
                ll += p["parameter"].loglikelihood_numerical_value(
                    vector_configuration[i]
                )
        return ll

    def check_validity(self, configuration: dict) -> bool:
        """Test whether the provided configuration is complete and valid.

        It checks wheter all parameters have a valid value or are not active.
        A warning is shown describing the issue.

        Args:
          configuration: The configuration to check.

        Returns:
            `True` if configuration is valid, `False` if not.
        """
        for n, p in self._parameters.items():
            v = configuration.get(n, None)
            if v is None:
                if p["condition"](configuration):
                    warnings.warn(
                        f"Parameter {n} should be active, "
                        + "but does not have a value assigned.\n"
                        + f"Full configuration: {configuration}",
                        RuntimeWarning,
                    )
                    return False
            else:
                if not p["condition"](configuration):
                    warnings.warn(
                        f"Parameter {n} = {v} should not be active!\n"
                        + f"Full configuration: {configuration}",
                        RuntimeWarning,
                    )
                    return False
                if not p["parameter"].check_value(v):
                    warnings.warn(
                        f"Parameter {n} = {v} is not a valid assignment!\n"
                        + f"Full configuration: {configuration}",
                        RuntimeWarning,
                    )
                    return False
        return True

    def remove_inactive(self, configuration: dict) -> dict:
        """Identify and remove parameters that should not have a value because
        their condition is unsatisfied.

        Args:
            configuration: Configuration to check.

        Returns:
            Cleaned configuration.
        """
        for n, p in self._parameters.items():
            if not p["condition"](configuration):
                configuration.pop(n, None)
        return configuration

    def to_numerical(self, configuration: dict) -> np.ndarray:
        self.remove_inactive(configuration)

        for name, value in self._constants.items():
            if name not in configuration:
                raise ValueError(
                    f"Configuration does not contain contant `{name} == {value}`!"
                )
            if configuration[name] != value:
                raise ValueError(
                    f"Constant parameter {name} has value {configuration[name]}, "
                    + f"but should be {value}!"
                )

        vec = np.zeros(len(self._parameters), dtype=float)
        for i, (n, p) in enumerate(self._parameters.items()):
            v = configuration.get(n, None)
            vec[i] = p["parameter"].val2num(v)
        return vec

    def val2num(self, configuration: dict) -> np.ndarray:
        """
        Attention:
            Deprecated. Use [ParameterSpace.to_numerical]\
            [parameterspace.parameterspace.ParameterSpace.to_numerical].

        Args:
            configuration: Configuration to convert.

        Raises:
            ValueError:  If configuration contains invalid names or values.

        Returns:
            Vector representation of the configuration.

        """
        warnings.warn(
            "Method ParameterSpace.val2num is deprecated and will be removed in "
            + "the future! Please use ParameterSpace.to_numerical instead.",
            category=DeprecationWarning,
        )
        return self.to_numerical(configuration)

    def from_numerical(self, vector: np.ndarray) -> dict:
        conf = {}
        for i, (n, p) in enumerate(self._parameters.items()):
            if not np.isnan(vector[i]):
                conf[n] = p["parameter"].num2val(vector[i])
        conf.update(self._constants)
        return self.remove_inactive(conf)

    def num2val(self, vector: np.ndarray) -> dict:
        """
        Attention:
            Deprecated. Use [ParameterSpace.from_numerical]\
            [parameterspace.parameterspace.ParameterSpace.from_numerical].

        Args:
            vector: Numerical vector representation of a configuration.

        Returns:
            Dictionary representation of the input configuration.

        """
        warnings.warn(
            "Method ParameterSpace.num2val is deprecated and will be removed in the "
            + "future! Please use ParameterSpace.from_numerical instead.",
            category=DeprecationWarning,
        )
        return self.from_numerical(vector)

    def has_conditions(self) -> bool:
        """Check if any of the parameters in the current `ParameterSpace` is
        conditioned on others.

        Returns:
            `True` of conditions exists, `False` if not.
        """
        for p in self:
            if not p["condition"].empty():
                return True
        return False

    def get_continuous_bounds(self) -> List[Tuple]:
        """Return the ParamerSpace specific bounds if it is purely continuous.

        Raises:
            ValueError: If `ParameterSpace` contains non-continuous parameters.

        Returns:
            Continuous bounds for this benchmark to easily run GP
        """
        bounds = []
        for n in self.get_parameter_names():
            p = self[n]["parameter"]
            if not p.is_continuous:
                raise ValueError(
                    "Parameterspace contains non-continuous parameter:\n{}".format(p)
                )
            bounds.append(tuple(p.get_numerical_bounds()))
        return bounds

    def to_latex_table(self, name_dict: Optional[dict] = None) -> str:
        """Construct LaTeX string for a table that shows the `ParameterSpace`
        with names, bounds and transformations.

        Args:
            name_dict: Mapping of parameter name in `ParameterSpace` to name
                to be rendered in LaTeX table.

        Returns:
            LaTeX table represenation
        """
        try:
            from num2tex import configure as num2tex_configure
            from num2tex import num2tex
        except ImportError as e:
            raise RuntimeError(
                "To use this functionality, please install num2tex."
            ) from e

        num2tex_configure(exp_format="cdot")

        name_dict = {} if name_dict is None else name_dict

        latex_strs = [
            "\\begin{tabular}{c c c c c}",
            "\\hline",
            "Parameter Name & Type & Values & Transformation & Prior \\\\",
            "\\hline",
        ]

        for parameter_name in self._parameters:
            parameter = self._parameters[parameter_name]["parameter"]

            name_str = name_dict.get(parameter_name, parameter.name)
            transformation_name = type(parameter._transformation).__name__
            prior_name = type(parameter._prior).__name__

            if isinstance(parameter, ps.IntegerParameter):
                type_str = "Integer"
                values_str = "$[{0[0]}, {0[1]}]$".format(parameter.bounds)
                transformation_str = "Log" if "Log" in transformation_name else ""
                prior_str = prior_name

            if isinstance(parameter, ps.ContinuousParameter):
                type_str = "Float"
                prior_str = prior_name

                if "Log" in transformation_name:
                    transformation_str = "Log"
                    values_str = "$[{}, {}]$".format(
                        num2tex(parameter.bounds[0], precision=2),
                        num2tex(parameter.bounds[1], precision=2),
                    )
                else:
                    transformation_str = " "
                    values_str = "$[{0[0]}, {0[1]}]$".format(parameter.bounds)

            if isinstance(parameter, ps.CategoricalParameter):
                type_str = "Categorical"
                values_str = "[" + ", ".join(parameter.values) + "]"
                transformation_str = " "
                prior_probs = parameter._prior.probabilities
                prior_str = (
                    "["
                    + ",".join(map(lambda p: "{:3.2f}".format(p), prior_probs))
                    + "]"
                )

            latex_strs.append(
                " & ".join(
                    [name_str, type_str, values_str, transformation_str, prior_str]
                )
                + "\\\\"
            )

        latex_strs.extend(["\\hline", "\\end{tabular}"])
        return "\n".join(latex_strs)
