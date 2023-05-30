"""Initialize a `ParameterSpace` from a `ConfigSpace` JSON dictionary."""

from typing import Dict, List, Optional, Tuple

import numpy as np

import parameterspace as ps
from parameterspace.condition import Condition
from parameterspace.utils import is_valid_python_variable_name, verify_lambda


def _escape_parameter_name(name: str) -> str:
    """Replace colons, dashes, dots and spaces with an underscore and add an underscore
    suffix to reserved Python words.
    """
    _name = name.replace(":", "_").replace("-", "_").replace(".", "_").replace(" ", "_")

    if not is_valid_python_variable_name(_name):
        _name = f"{_name}_"

    if not is_valid_python_variable_name(_name):
        raise ValueError(
            f'Failed to transform "{name}" into a valid parameter name, '
            + f'ended up with "{_name}".'
        )

    return _name


def _get_condition(
    conditions: List[dict], configspace_parameter_name: str
) -> Optional[Condition]:
    """Construct a lambda function that can be used as a ParameterSpace condition from a
    ConfigSpace conditions list given a specific target parameter name.

    NOTE: The `configspace_parameter_name` here needs to match the original name in
    `ConfigSpace`, not the one transformed with `_escape_parameter_name`.
    """
    condition = Condition()

    varnames = []
    function_texts = []
    for cond in conditions:
        if cond["child"] == configspace_parameter_name:
            parent = _escape_parameter_name(cond["parent"])
            varnames.append(parent)
            # The representation is used because it quotes strings.
            if cond["type"] == "IN":
                function_texts.append(f"{parent} in {tuple(cond['values'])}")
            elif cond["type"] == "EQ":
                function_texts.append(f"{parent} == {repr(cond['value'])}")
            elif cond["type"] == "NEQ":
                function_texts.append(f"{parent} != {repr(cond['value'])}")
            elif cond["type"] == "GT":
                function_texts.append(f"{parent} > {repr(cond['value'])}")
            elif cond["type"] == "LT":
                function_texts.append(f"{parent} < {repr(cond['value'])}")
            else:
                raise NotImplementedError(f"Unsupported condition type {cond['type']}")

    if not varnames:
        return condition

    function_text = " and ".join(function_texts)
    verify_lambda(variables=varnames, body=function_text)
    # pylint: disable=eval-used
    condition_function = eval(f"lambda {', '.join(varnames)}: {function_text}")
    # pylint: enable=eval-used

    condition.function_texts.append(function_text)
    condition.varnames.append(varnames)
    condition.all_varnames |= set(varnames)
    condition.functions.append(condition_function)

    return condition


def _convert_for_normal_parameter(
    log: bool, lower: Optional[float], upper: Optional[float], mu: float, sigma: float
) -> Tuple[float, float, float, float]:
    """Convert bounds and prior mean/std from `ConfigSpace` parameter dictionary with
    normal prior to `ParameterSpace` compatible values.

    Args:
        log: Are we on a log scale?
        lower: Optional lower bound in the original space (required when `log=True`)
        upper: Optional upper bound in the original space (required when `log=True`)
        mu: Mean of the `ConfigSpace` normal distribution
        sigma: Standard deviation of the `ConfigSpace` normal distribution

    Returns:
        Transformed lower bound, upper bound, mean and standard deviation

    Raises:
        Value error when log is True but bounds are missing.
    """
    if lower is None or upper is None:
        if log:
            raise ValueError(
                "Please provide bounds, when using a log transform with a normal prior."
            )
        lower = mu - 4 * sigma
        upper = mu + 4 * sigma

    if log:
        log_upper, log_lower = np.log(upper), np.log(lower)
        log_interval_size = log_upper - log_lower
        mean = (mu - log_lower) / log_interval_size
        std = sigma / log_interval_size
    else:
        interval_size = upper - lower
        mean = (mu - lower) / interval_size
        std = sigma / interval_size

    return lower, upper, mean, std


def parameterspace_from_configspace_dict(
    configspace_dict: dict,
) -> Tuple[ps.ParameterSpace, Dict[str, str]]:
    """Create `ParameterSpace` instance from a `ConfigSpace` JSON dictionary.

    Note, that `ParameterSpace` does not support regular, non-truncated normal priors
    and will thus translate an unbounded normal prior to a normal truncated at +/- 4
    sigma. Also, constant parameters are represented as categoricals with a single value
    that are fixed to said value.

    Args:
        configspace_dict: The dictionary based on a `ConfigSpace` JSON representation.

    Returns:
        A `ParameterSpace` instance.
        A mapping between parameter names that were changed for compatibility reasons.

    Raises:
        NotImplementedError in case a given parameter type or configuration is not
        supported.
    """
    space = ps.ParameterSpace()
    names: Dict[str, str] = {}

    for param_dict in configspace_dict["hyperparameters"]:
        param_name = _escape_parameter_name(param_dict["name"])
        if param_dict["name"] != param_name:
            names[param_name] = param_dict["name"]
            names[param_dict["name"]] = param_name

        condition = _get_condition(configspace_dict["conditions"], param_dict["name"])
        if param_dict["type"] == "uniform_int":
            space._parameters[param_name] = {
                "parameter": ps.IntegerParameter(
                    name=param_name,
                    bounds=(param_dict["lower"], param_dict["upper"]),
                    transformation="log" if param_dict["log"] else None,
                ),
                "condition": condition,
            }

        elif param_dict["type"] == "categorical":
            space._parameters[param_name] = {
                "parameter": ps.CategoricalParameter(
                    name=param_name,
                    values=param_dict["choices"],
                    prior=param_dict.get("weights", None),
                ),
                "condition": condition,
            }

        elif param_dict["type"] in ["constant", "unparametrized"]:
            space._parameters[param_name] = {
                "parameter": ps.CategoricalParameter(
                    name=param_name,
                    values=[param_dict["value"]],
                ),
                "condition": condition,
            }
            space.fix(**{param_name: param_dict["value"]})

        elif param_dict["type"] in ["normal_float", "normal_int"]:
            parameter_class = (
                ps.ContinuousParameter
                if param_dict["type"] == "normal_float"
                else ps.IntegerParameter
            )
            lower_bound, upper_bound, mean, std = _convert_for_normal_parameter(
                log=param_dict["log"],
                lower=param_dict.get("lower", None),
                upper=param_dict.get("upper", None),
                mu=param_dict["mu"],
                sigma=param_dict["sigma"],
            )
            space._parameters[param_name] = {
                "parameter": parameter_class(
                    name=param_name,
                    bounds=(lower_bound, upper_bound),
                    prior=ps.priors.TruncatedNormal(mean=mean, std=std),
                    transformation="log" if param_dict["log"] else None,
                ),
                "condition": condition,
            }

        elif param_dict["type"] == "uniform_float":
            space._parameters[param_name] = {
                "parameter": ps.ContinuousParameter(
                    name=param_name,
                    bounds=(param_dict["lower"], param_dict["upper"]),
                    transformation="log" if param_dict["log"] else None,
                ),
                "condition": condition,
            }

        elif param_dict["type"] == "ordinal":
            space._parameters[param_name] = {
                "parameter": ps.OrdinalParameter(
                    name=param_name,
                    values=param_dict["sequence"],
                ),
                "condition": condition,
            }

        else:
            raise NotImplementedError(f"Unsupported type {param_dict['type']}")

    return space, names
