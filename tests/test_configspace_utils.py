import json

from parameterspace.configspace_utils import parameterspace_from_configspace_dict
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
      "name": "lambda",
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


def test_parameterspace_from_configspace_with_conditions():
    cs_dict = json.loads(CS_CONDITIONS_JSON)
    space = parameterspace_from_configspace_dict(cs_dict)
    assert len(space) == 7
    assert space.has_conditions()
    assert space.check_validity(
        {
            "alpha": 1.0,
            "booster": "gbtree",
            "lambda": 1.0,
            "nrounds": 122,
            "repl": 6,
            "max_depth": 3,
        }
    )
    assert isinstance(
        space._parameters["nrounds"]["parameter"]._transformation,
        LogZeroOneIntegerTransformation,
    )

    alpha = cs_dict["hyperparameters"][0]
    assert alpha["name"] == "alpha"
    bounds = [alpha["lower"], alpha["upper"]]
    assert list(space._parameters["alpha"]["parameter"].bounds) == bounds

    booster = cs_dict["hyperparameters"][1]
    assert booster["name"] == "booster"
    assert space._parameters["booster"]["parameter"].values == booster["choices"]

    _s = space.copy()
    _s.fix(booster="dart")
    assert len(_s.sample()) == 7

    _s = space.copy()
    _s.fix(booster="gblinear")
    assert len(_s.sample()) == 5


def test_parameterspace_from_configspace_with_priors():
    assert False
