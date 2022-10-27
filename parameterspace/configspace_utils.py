"""Initialize a `ParameterSpace` from a `ConfigSpace` JSON dictionary."""

from typing import List, Optional

import parameterspace as ps
from parameterspace.condition import Condition
from parameterspace.utils import verify_lambda


def _escape_parameter_name(name: str) -> str:
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
                # Using the valid Python "a in [2,3,4]" is not supported, hence use or
                function_texts.append(
                    " or ".join([f"{parent} == {v.__repr__()}" for v in cond["values"]])
                )
            elif cond["type"] == "EQ":
                function_texts.append(f"{parent} == {cond['value'].__repr__()}")
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

        elif param_dict["type"] == "constant":
            # FIXME: Is there a better alternative to a constant than a one choice cat?
            space._parameters[param_name] = {
                "parameter": ps.CategoricalParameter(
                    name=param_name,
                    values=[param_dict["value"]],
                ),
                "condition": condition,
            }

        elif param_dict["type"] == "normal_float":
            lower_bound = param_dict["mu"] - 4 * param_dict["sigma"]
            upper_bound = param_dict["mu"] + 4 * param_dict["sigma"]
            if param_dict["log"]:
                lower_bound = max(lower_bound, 1e-24)

            space._parameters[param_name] = {
                "parameter": ps.ContinuousParameter(
                    name=param_name,
                    bounds=(lower_bound, upper_bound),
                    prior=ps.priors.TruncatedNormal(
                        mean=param_dict["mu"], std=param_dict["sigma"]
                    ),
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
