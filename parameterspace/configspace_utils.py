"""Initialize a `ParameterSpace` from a `ConfigSpace` JSON dictionary."""

from typing import List, Optional

import numpy as np

import parameterspace as ps
from parameterspace.condition import Condition
from parameterspace.utils import verify_lambda


def _escape_parameter_name(name: str) -> str:
    """Replace colons with underscores.

    Colons are incompatible as ParameterSpace parameter names.
    """
    return name.replace(":", "_")


def _get_condition(conditions: List[dict], parameter_name: str) -> Optional[Condition]:
    """Construct a lambda function that can be used as a ParameterSpace condition from a
    ConfigSpace conditions list given a specific target parameter name.
    """
    condition = Condition()

    varnames = []
    function_texts = []
    for cond in conditions:
        if cond["child"] == parameter_name:
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


def parameterspace_from_configspace_dict(configspace_dict: dict) -> ps.ParameterSpace:
    space = ps.ParameterSpace()

    for param_dict in configspace_dict["hyperparameters"]:
        param_name = _escape_parameter_name(param_dict["name"])
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
            if (
                param_dict.get("lower", None) is None
                or param_dict.get("upper", None) is None
            ):
                if param_dict["log"]:
                    raise ValueError(
                        "Please provide bounds, when using a log transform with a "
                        + "normal prior."
                    )

                lower_bound = param_dict["mu"] - 4 * param_dict["sigma"]
                upper_bound = param_dict["mu"] + 4 * param_dict["sigma"]
            else:
                lower_bound, upper_bound = param_dict["lower"], param_dict["upper"]

            parameter_class = (
                ps.ContinuousParameter
                if param_dict["type"] == "normal_float"
                else ps.IntegerParameter
            )

            if param_dict["log"]:
                log_upper, log_lower = np.log(upper_bound), np.log(lower_bound)
                log_interval_size = log_upper - log_lower
                mean = (param_dict["mu"] - log_lower) / log_interval_size
                std = param_dict["sigma"] / log_interval_size
            else:
                interval_size = upper_bound - lower_bound
                mean = (param_dict["mu"] - lower_bound) / interval_size
                std = param_dict["sigma"] / interval_size

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

    return space
