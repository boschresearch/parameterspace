# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import numpy as np
import pytest
from parameterspace.condition import Condition


def test_condition_lambda_fn():
    c = Condition(lambda x1, x2: x1 - x2 > 0)

    assert c.all_varnames == set(["x1", "x2"])
    assert c(dict(x1=2, x2=1))
    assert c(dict(x2=1, x1=2))
    assert not c(dict(x1=1, x2=2))
    with pytest.raises(ValueError):
        c(dict(x1=5))


def test_condition_regular_fn():
    def condition_fn(x2, x1=5):
        return x1 - x2 > 0

    with pytest.raises(RuntimeError):
        c = Condition(condition_fn)


def test_equal():
    c1 = Condition(lambda x1: x1 > 0)
    c2 = Condition(lambda x1: x1 < 0)
    c3 = Condition(lambda x2: x2 > 0)
    c4 = Condition(lambda x1: x1 == 0).merge(c1)
    assert c1 != c2
    assert c1 != c3
    assert c1 != c4


def test_merge():
    c1 = Condition(lambda x1, x2: x1 - x2 > 0)
    c2 = Condition(lambda x3, x2: x2 - x3 > 0)
    c3 = Condition(lambda x1, x2: x1 - x2 > 0).merge(c2)

    refs = [
        ({"x1": 3, "x2": 2, "x3": 1}, [True, True]),
        ({"x1": 3, "x2": 3, "x3": 1}, [False, True]),
        ({"x1": 1, "x2": 2, "x3": 3}, [False, False]),
        ({"x1": 3, "x2": 2, "x3": 3}, [True, False]),
        ({"x1": None, "x2": 2, "x3": 3}, [False, False]),
    ]

    assert len(c3.functions) == 2
    assert len(c3.varnames) == 2

    for conf, vals in refs:
        assert c1(conf) == vals[0]
        assert c2(conf) == vals[1]
        assert c3(conf) == (vals[0] and vals[1])


def test_no_init_args():
    c = Condition()
    assert c({})
    assert len(c.all_varnames) == 0


def test_repr():
    c = Condition(lambda x1, x2: x1 - x2 > 0)
    s = str(c)
    assert "x1 - x2 > 0" in s

    c = Condition()
    s = str(c)
    assert "No conditions!" in s


def test_empty():
    c = Condition()
    assert c.empty()

    c = Condition(lambda x1, x2: x1 - x2 > 0)
    assert not c.empty()


def test_merge_empty():
    c1 = Condition()
    assert c1.empty()

    c2 = Condition(lambda x1: x1 > 0)
    assert not c2.empty()

    for x in np.linspace(-1, 1, 16):
        assert c2({"x1": x}) == (x > 0)


def test_from_to_dict(n_samples=64):
    c1 = Condition(lambda x1, x2: x1 - x2 > 0)
    c2 = Condition(lambda x3, x2: x2 - x3 > 0).merge(c1)
    c3 = Condition(lambda x1, x3: x1 - x3 > 0).merge(c2)

    random_inputs = np.random.rand(n_samples, 3)

    json_list = c3.to_dict()
    json.dumps(json_list)
    c4 = Condition.from_dict(json_list)

    assert c3 == c4
    for x1, x2, x3 in random_inputs:
        assert c3(dict(x1=x1, x2=x2, x3=x3)) == c4(dict(x1=x1, x2=x2, x3=x3))

    # make sure a deserialized condition can be serialized again
    json_list2 = c4.to_dict()
    assert json_list == json_list2

    malicious_json = copy.deepcopy(json_list)
    malicious_json["0"]["function_text"] = "eval()"

    with pytest.raises(RuntimeError):
        Condition.from_dict(malicious_json)


if __name__ == "__main__":
    pytest.main(("--pdb %s" % __file__).split())
