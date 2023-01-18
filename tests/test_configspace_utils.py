import json

import numpy as np
import pytest
from ConfigSpace import (
    ConfigurationSpace,
    EqualsCondition,
    Float,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    Normal,
    NotEqualsCondition,
)
from ConfigSpace.read_and_write import json as cs_json
from scipy.stats import truncnorm as scipy_truncnorm

from parameterspace.configspace_utils import parameterspace_from_configspace_dict
from parameterspace.priors.categorical import Categorical as CategoricalPrior
from parameterspace.priors.truncated_normal import (
    TruncatedNormal as TruncatedNormalPrior,
)
from parameterspace.transformations.log_zero_one import (
    LogZeroOneInteger as LogZeroOneIntegerTransformation,
)

CS_CONDITIONS_JSON = """{
  "hyperparameters": [
    {
      "name": "alpha",
      "type": "uniform_float",
      "log": true,
      "lower": 0.001,
      "upper": 1095.0,
      "default": 1.0
    },
    {
      "name": "booster",
      "type": "categorical",
      "choices": [
        "gblinear",
        "gbtree",
        "dart"
      ],
      "default": "gblinear",
      "probabilities": null
    },
    {
      "name": "lambda_",
      "type": "uniform_float",
      "log": true,
      "lower": 0.0009118819655545162,
      "upper": 1096.6331584284585,
      "default": 1.0
    },
    {
      "name": "nrounds",
      "type": "uniform_int",
      "log": true,
      "lower": 8,
      "upper": 2980,
      "default": 122
    },
    {
      "name": "repl",
      "type": "uniform_int",
      "log": false,
      "lower": 1,
      "upper": 10,
      "default": 6
    },
    {
      "name": "max_depth",
      "type": "uniform_int",
      "log": false,
      "lower": 3,
      "upper": 10,
      "default": 3
    },
    {
      "name": "rate_drop",
      "type": "uniform_float",
      "log": false,
      "lower": 0.0,
      "upper": 1.0,
      "default": 0.2
    }
  ],
  "conditions": [
    {
      "child": "max_depth",
      "parent": "booster",
      "type": "IN",
      "values": [
        "dart",
        "gbtree"
      ]
    },
    {
      "child": "rate_drop",
      "parent": "booster",
      "type": "EQ",
      "value": "dart"
    }
  ],
  "forbiddens": [],
  "python_module_version": "0.4.18",
  "json_format_version": 0.2
}"""


def _cs_to_dict(cs: ConfigurationSpace) -> dict:
    return json.loads(cs_json.write(cs))


def test_conditions_and_log_transform():
    cs_dict = json.loads(CS_CONDITIONS_JSON)
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 7
    assert space.has_conditions()
    assert space.check_validity(
        {
            "alpha": 1.0,
            "booster": "gbtree",
            "lambda_": 1.0,
            "nrounds": 122,
            "repl": 6,
            "max_depth": 3,
        }
    )
    assert isinstance(
        space._parameters["nrounds"]["parameter"]._transformation,
        LogZeroOneIntegerTransformation,
    )

    _s = space.copy()
    _s.fix(booster="dart")
    assert len(_s.sample()) == 7

    _s = space.copy()
    _s.fix(booster="gblinear")
    assert len(_s.sample()) == 5

    alpha = cs_dict["hyperparameters"][0]
    assert alpha["name"] == "alpha"
    bounds = [alpha["lower"], alpha["upper"]]
    assert list(space._parameters["alpha"]["parameter"].bounds) == bounds

    booster = cs_dict["hyperparameters"][1]
    assert booster["name"] == "booster"
    assert space._parameters["booster"]["parameter"].values == booster["choices"]


def test_continuous_with_normal_prior():
    cs_dict = json.loads(
        """{
          "name": "myspace",
          "hyperparameters": [
            {
              "name": "p",
              "type": "normal_float",
              "log": false,
              "mu": 8.0,
              "sigma": 10.0,
              "default": 6.2
            }
          ],
          "conditions": [],
          "forbiddens": [],
          "python_module_version": "0.6.0",
          "json_format_version": 0.4
        }
        """
    )
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 1

    param = space.get_parameter_by_name("p")["parameter"]
    assert isinstance(param._prior, TruncatedNormalPrior)

    samples = np.array([space.sample()["p"] for _ in range(10_000)])
    assert abs(samples.mean() - cs_dict["hyperparameters"][0]["mu"]) < 0.1
    assert abs(samples.std() - cs_dict["hyperparameters"][0]["sigma"]) < 0.2


def test_continuous_with_log_normal_prior_and_no_bounds_raises():
    cs = ConfigurationSpace(
        space={
            "p": Float(
                "p",
                default=0.1,
                log=True,
                distribution=Normal(1.0, 0.6),
            ),
        },
    )
    with pytest.raises(ValueError):
        parameterspace_from_configspace_dict(_cs_to_dict(cs))


def test_continuous_with_log_normal_prior():
    mu = 1.0
    sigma = 0.6
    cs_dict = json.loads(
        f"""{{
          "name": "myspace",
          "hyperparameters": [
            {{
              "name": "p",
              "type": "normal_float",
              "lower": 1e-5,
              "upper": 1e-1,
              "log": true,
              "mu": {mu},
              "sigma": {sigma},
              "default": 1.1
            }}
          ],
          "conditions": [],
          "forbiddens": [],
          "python_module_version": "0.6.0",
          "json_format_version": 0.4
        }}
        """
    )
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 1

    param = space.get_parameter_by_name("p")["parameter"]
    assert isinstance(param._prior, TruncatedNormalPrior)

    samples = np.array([space.sample()["p"] for _ in range(10_000)])

    a, b = (np.log(param.bounds) - mu) / sigma
    expected_mean = scipy_truncnorm.stats(a, b, loc=mu, scale=sigma, moments="m")
    assert abs(np.log(samples).mean() - expected_mean) < 0.1

    expected_var = scipy_truncnorm.stats(a, b, loc=mu, scale=sigma, moments="v")
    assert abs(np.log(samples).var() - expected_var) < 0.1


def test_integer_with_normal_prior():
    cs_dict = json.loads(
        """{
          "name": "myspace",
          "hyperparameters": [
            {
              "name": "p",
              "type": "normal_int",
              "log": false,
              "mu": 8.0,
              "sigma": 5.0,
              "default": 2
            }
          ],
          "conditions": [],
          "forbiddens": [],
          "python_module_version": "0.6.0",
          "json_format_version": 0.4
        }
        """
    )
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 1

    param = space.get_parameter_by_name("p")["parameter"]
    assert isinstance(param._prior, TruncatedNormalPrior)

    samples = np.array([space.sample()["p"] for _ in range(10_000)])

    assert abs(samples.mean() - cs_dict["hyperparameters"][0]["mu"]) < 0.1
    assert abs(samples.std() - cs_dict["hyperparameters"][0]["sigma"]) < 0.3


def test_categorical_with_custom_probabilities():
    cs_dict = json.loads(
        """{
          "name": "myspace",
          "hyperparameters": [
            {
              "name": "c",
              "type": "categorical",
              "choices": [
                "red",
                "green",
                "blue"
              ],
              "default": "blue",
              "weights": [
                2,
                1,
                1
              ]
            }
          ],
          "conditions": [],
          "forbiddens": [],
          "python_module_version": "0.6.0",
          "json_format_version": 0.4
        }"""
    )
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 1

    param = space.get_parameter_by_name("c")["parameter"]
    assert isinstance(param._prior, CategoricalPrior)
    reference_weights = np.array(cs_dict["hyperparameters"][0]["weights"])
    assert np.all(
        param._prior.probabilities == reference_weights / reference_weights.sum()
    )


def test_equals_condition():
    cs = ConfigurationSpace({"a": [1, 2, 3], "b": (1.0, 8.0)})
    cond = EqualsCondition(cs["b"], cs["a"], 1)
    cs.add_condition(cond)

    space = parameterspace_from_configspace_dict(_cs_to_dict(cs))
    assert len(space) == 2

    _s = space.copy()
    _s.fix(a=1)
    assert len(_s.sample()) == 2

    _s = space.copy()
    _s.fix(a=2)
    assert len(_s.sample()) == 1


def test_not_equals_condition():
    cs = ConfigurationSpace({"a": [1, 2, 3], "b": (1.0, 8.0)})
    cond = NotEqualsCondition(cs["b"], cs["a"], 1)
    cs.add_condition(cond)

    space = parameterspace_from_configspace_dict(_cs_to_dict(cs))
    assert len(space) == 2

    _s = space.copy()
    _s.fix(a=2)
    assert len(_s.sample()) == 2

    _s = space.copy()
    _s.fix(a=1)
    assert len(_s.sample()) == 1


def test_less_than_condition():
    cs = ConfigurationSpace({"a": (0, 10), "b": (1.0, 8.0)})
    cond = LessThanCondition(cs["b"], cs["a"], 5)
    cs.add_condition(cond)

    space = parameterspace_from_configspace_dict(_cs_to_dict(cs))
    assert len(space) == 2

    _s = space.copy()
    _s.fix(a=4)
    assert len(_s.sample()) == 2

    _s = space.copy()
    _s.fix(a=6)
    assert len(_s.sample()) == 1


def test_greater_than_condition():
    cs = ConfigurationSpace({"a": (0, 10), "b": (1.0, 8.0)})
    cond = GreaterThanCondition(cs["b"], cs["a"], 5)
    cs.add_condition(cond)

    space = parameterspace_from_configspace_dict(_cs_to_dict(cs))
    assert len(space) == 2

    _s = space.copy()
    _s.fix(a=6)
    assert len(_s.sample()) == 2

    _s = space.copy()
    _s.fix(a=4)
    assert len(_s.sample()) == 1


def test_in_condition():
    cs = ConfigurationSpace({"a": (0, 10), "b": (1.0, 8.0)})
    cond = InCondition(cs["b"], cs["a"], [1, 2, 3, 4])
    cs.add_condition(cond)

    space = parameterspace_from_configspace_dict(_cs_to_dict(cs))
    assert len(space) == 2

    _s = space.copy()
    _s.fix(a=2)
    assert len(_s.sample()) == 2

    _s = space.copy()
    _s.fix(a=5)
    assert len(_s.sample()) == 1
