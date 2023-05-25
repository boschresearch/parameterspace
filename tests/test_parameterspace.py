# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import math

import dill
import numpy as np
import pytest
from scipy.stats import norm, truncnorm

import parameterspace as ps
from parameterspace.parameters.categorical import CategoricalParameter
from parameterspace.parameters.continuous import ContinuousParameter
from parameterspace.parameters.integer import IntegerParameter
from parameterspace.parameters.ordinal import OrdinalParameter
from parameterspace.parameterspace import ParameterSpace
from parameterspace.priors.truncated_normal import TruncatedNormal


def test_simple_space():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    names = space.get_parameter_names()

    assert "p1" in names
    assert "p2" in names
    assert space["p1"]["parameter"] == p1
    assert space["p1"]["condition"]()
    assert space["p2"]["parameter"] == p2
    assert space["p2"]["condition"]()
    assert len(space) == 2

    names = [p["parameter"].name for p in space]

    assert names[0] == "p1"
    assert names[1] == "p2"
    assert not space.has_conditions()

    assert space.get_parameter_by_name("p1")["parameter"] == p1
    with pytest.raises(KeyError):
        space.get_parameter_by_name("p3")

    sample1 = space.sample(rng=np.random.default_rng(42))
    sample2 = space.sample(rng=np.random.default_rng(42))

    assert sample1 == sample2


def test_seeded_parameterspace():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter(
        "p2", (1e-3, 1e3), prior=TruncatedNormal(0.5, 1), transformation="log"
    )
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space1 = ParameterSpace(seed=42)
    space1.add(p1)
    space1.add(p2)
    space1.add(p3)
    space1.add(p4)

    space2 = ParameterSpace(seed=42)
    space2.add(p1)
    space2.add(p2)
    space2.add(p3)
    space2.add(p4)

    space3 = ParameterSpace()
    space3.add(p1)
    space3.add(p2)
    space3.add(p3)
    space3.add(p4)

    sample1 = space1.sample()
    sample2 = space2.sample()
    sample3 = space3.sample()
    assert sample1 == sample2, "Equally seeded spaces should produce the same sample"
    assert sample1 != sample3, "An unseeded space should produce a different sample"

    space2_reloaded = ParameterSpace.from_dict(space2.to_dict())
    reloaded_next = space2_reloaded.sample()
    original_next = space2.sample()
    assert (
        reloaded_next == original_next
    ), "To and from dictionary serialization should not interfere with the seeding."


def test_remove_parameter_no_conditions():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    assert set(["p1", "p2"]) == set(space.get_parameter_names())

    space.remove("p1")
    assert set(["p2"]) == set(space.get_parameter_names())

    space.remove("p2")
    assert len(space.get_parameter_names()) == 0

    with pytest.raises(KeyError):
        space.remove("p1")


def test_remove_parameter_with_conditions():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))
    p3 = ContinuousParameter("p3", (0, 5))
    space = ParameterSpace()

    space.add(p1)
    space.add(p2, lambda p1: p1 > 0)
    space.add(p3, lambda p2: p2 > 2.5)

    assert set(["p1", "p2", "p3"]) == set(space.get_parameter_names())

    with pytest.raises(RuntimeError):
        space.remove("p1")

    with pytest.raises(RuntimeError):
        space.remove("p2")

    assert set(["p1", "p2", "p3"]) == set(space.get_parameter_names())

    space.remove("p3")
    assert space.has_conditions()

    space.remove("p2")
    assert not space.has_conditions()

    space.remove("p1")
    assert len(space.get_parameter_names()) == 0


def test_remove_hierarchical_parameter_goes_unnoticed():
    s1 = ParameterSpace()
    s1.add(IntegerParameter("p1", (-5, 5)))

    s2 = ParameterSpace()
    s2.add(IntegerParameter("p3", (-5, 5)))

    s2.add(s1)

    # NOTE: s2, by design, won't ever know about this
    s1.remove("p1")
    assert "p1" in s2.get_parameter_names()


def test_sampling_from_invalid_partial_assignment():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    with pytest.raises(ValueError):
        space.sample(partial_configuration={"p1": -6})


def test_sampling_from_partial_assignment():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    for _ in range(16):
        c = space.sample(partial_configuration={"p1": 0})
        space.check_validity(c)
        assert c["p1"] == 0


def test_parameter_name_exists():
    # adding a parameter with the same name than another parameter
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p1", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    with pytest.raises(ValueError):
        space.add(p2)


def test_checking_invalid_configs():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2, lambda p1: p1 >= 0)

    config1 = dict(p1=0, p2=-1)

    with pytest.warns(RuntimeWarning):
        assert not space.check_validity(config1)

    config2 = dict(p1=0, p2=None)
    with pytest.warns(RuntimeWarning):
        assert not space.check_validity(config2)

    config3 = dict(p1=-1, p2=2)
    with pytest.warns(RuntimeWarning):
        assert not space.check_validity(config3)

    config4 = dict(p1=0, p2=0)
    space.check_validity(config4)


def test_hierarchical_space():
    p1 = CategoricalParameter("p1", [True, False])
    p2 = CategoricalParameter("p2", [True, False])
    p3 = CategoricalParameter("p3", [True, False])
    p4 = CategoricalParameter("p4", [True, False])
    p5 = CategoricalParameter("p5", [True, False])

    top_level_space = ParameterSpace()
    top_level_space.add(p1)
    top_level_space.add(p2, lambda p1: p1)

    subspace = ParameterSpace()
    subspace.add(p3)
    subspace.add(p4, lambda p3: not p3)
    subspace.add(p5, lambda p3: p3)

    top_level_space.add(subspace, lambda p2: p2)
    all_param_names = [p["parameter"].name for p in top_level_space]

    # reference configurations that should cover all possible combinations of satisfied
    # and unsatisfied conditions and whether those are valid or not.

    should_contain = dict(
        p1=lambda conf: True,
        p2=lambda conf: conf["p1"],
        p3=lambda conf: conf["p1"] and conf["p2"],
        p4=lambda conf: conf["p1"] and conf["p2"] and not conf["p3"],
        p5=lambda conf: conf["p1"] and conf["p2"] and conf["p3"],
    )

    for values in itertools.product([True, False, None], repeat=5):
        config = {f"p{i+1}": values[i] for i in range(4) if values[i] is not None}

        try:
            should = [should_contain[p](config) for p in all_param_names]
            does = [p in config for p in all_param_names]

            is_invalid = False

            for s, d in zip(should, does):
                if (s and not d) or (not s and d):
                    is_invalid = True

            if is_invalid:
                with pytest.warns(RuntimeWarning):
                    assert not top_level_space.check_validity(config)
            else:
                assert top_level_space.check_validity(config)

        except KeyError:
            # this KeyError occurs for ill-formed configurations with active parameters
            # missing which means it's a invalid configuration yielding a warning
            with pytest.warns(RuntimeWarning):
                assert not top_level_space.check_validity(config)


def test_loglikelihood_1d():
    p1 = ContinuousParameter("p1", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))

    s1 = ParameterSpace()
    s1.add(p1)

    ref_ll = s1.log_likelihood({"p1": 0})
    ref_ll_sps = norm.logpdf(0)

    ref = [({"p1": i}, norm.logpdf(i)) for i in range(-5, 6)]

    for c, ref_val in ref:
        assert np.allclose(
            s1.log_likelihood(c) - ref_ll, ref_val - ref_ll_sps, atol=1e-6
        )


def test_loglikelihood_2d_no_condition():
    p1 = ContinuousParameter("p1", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))
    p2 = ContinuousParameter("p2", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))

    s1 = ParameterSpace()
    s1.add(p1)
    s1.add(p2)

    # we will compare to a normal distribution, as the truncation is at 5 sigma, so
    # almost irrelevant.
    ref_ll = s1.log_likelihood({"p1": 0, "p2": 0})
    ref_ll_sps = 2 * norm.logpdf(0)

    ref = [
        ({"p1": -2.5, "p2": 5}, norm.logpdf(-2.5) + norm.logpdf(5)),
        ({"p1": 0.2, "p2": 1}, norm.logpdf(0.2) + norm.logpdf(1)),
        ({"p1": 1, "p2": 1}, norm.logpdf(1) + norm.logpdf(1)),
    ]

    for c, ref_val in ref:
        assert np.allclose(
            s1.log_likelihood(c) - ref_ll, ref_val - ref_ll_sps, atol=1e-6
        )


def test_loglikelihood_2d_with_condition():
    p1 = ContinuousParameter("p1", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))
    p2 = ContinuousParameter("p2", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))

    s1 = ParameterSpace()
    s1.add(p1)
    s1.add(p2, lambda p1: p1 >= 0)

    a, b = (np.array([-5.0, 5.0]) - 0) / 1
    sps_dist = truncnorm(a, b, loc=0.0, scale=1.0)

    ref = [
        ({"p1": 0, "p2": 0}, 2 * (sps_dist.logpdf(0))),
        ({"p1": 0.2, "p2": 1}, norm.logpdf(0.2) + norm.logpdf(1)),
        ({"p1": -2}, norm.logpdf(-2)),
        ({"p1": -2.5}, norm.logpdf(-2.5)),
    ]

    for c, ref_val in ref:
        assert np.allclose(s1.log_likelihood(c), ref_val, atol=1e-6)

    # test invalid config, too
    with pytest.warns(RuntimeWarning):
        assert np.isnan(s1.log_likelihood({"p1": -1, "p2": 0}))


def test_repr():
    p1 = ContinuousParameter("p1", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))
    p2 = ContinuousParameter("p2", (-5, 5), prior=TruncatedNormal(mean=0.5, std=0.1))

    s1 = ParameterSpace()
    s1.add(p1)
    s1.add(p2, lambda p1: p1 >= 0)

    s = str(s1)

    assert "p1" in s
    assert "p2" in s
    assert "p1 >= 0" in s


def test_conditional_space(num_samples=128):
    optimizer = CategoricalParameter("optimizer", ["Adam", "SGD"])

    lr_adam = ContinuousParameter(
        "lr_adam",
        bounds=[1e-5, 1e-3],
        transformation=ps.transformations.LogZeroOneFloat([1e-5, 1e-3]),
    )
    lr_sgd = ContinuousParameter(
        "lr_sgd",
        bounds=[1e-3, 1e-0],
        transformation="log",
        prior=ps.priors.TruncatedNormal(mean=0.5, std=0.2),
    )
    momentum_sgd = ContinuousParameter(
        "momentum", bounds=[0, 0.9], inactive_numerical_value=-1
    )

    space = ParameterSpace()
    space.add(optimizer)
    space.add(lr_adam, lambda optimizer: optimizer == "Adam")
    space.add(lr_sgd, lambda optimizer: optimizer == "SGD")
    space.add(momentum_sgd, lambda optimizer: optimizer == "SGD")

    for _ in range(num_samples):
        s = space.sample()
        assert s["optimizer"] in ["Adam", "SGD"]

        s_num = space.to_numerical(s)

        if optimizer == "Adam":
            assert np.isnan(s_num[2])
            assert s_num[3] == -1
        if optimizer == "SGD":
            assert np.isnan(s_num[1])

    assert space.has_conditions()


def test_num2val2num(num_samples=128):
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space = ParameterSpace()
    space.add(p1)
    space.add(p2)
    space.add(p3)
    space.add(p4)

    for _ in range(num_samples):
        sample = space.sample()
        num = space.to_numerical(sample)
        s2 = space.from_numerical(num)

        for k, s in sample.items():
            assert s == s2[k]


def test_get_continuous_bounds():
    p1 = ContinuousParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    s1 = ParameterSpace()
    s1.add(p1)
    s1.add(p2)

    assert s1.get_continuous_bounds() == [(0, 1), (0, 1)]

    p3 = IntegerParameter("p3", (-5, 5))

    s1.add(p3)

    with pytest.raises(ValueError):
        s1.get_continuous_bounds()


def test_to_latex_table():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (1e-3, 1e-1), transformation="log")
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space = ParameterSpace()
    space.add(p1)
    space.add(p2)
    space.add(p3)
    space.add(p4)

    name_dict = {
        "p1": "1st parameter name",
        "p2": "2nd parameter name",
        "p3": "3rd parameter name",
        "p4": "4th parameter name",
    }

    latex_str = space.to_latex_table(name_dict)

    for latex_name in name_dict.values():
        assert latex_name in latex_str


def test_from_dict_without_rng_state():
    space = ParameterSpace()
    space.add(IntegerParameter("p1", (-5, 5)))
    space_dict = space.to_dict()
    assert space_dict.pop("bit_generator_state") is not None
    ParameterSpace.from_dict(space_dict)


def test_to_from_dict(num_samples=128):
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter(
        "p2", (1e-3, 1e3), prior=TruncatedNormal(0.5, 1), transformation="log"
    )
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space1 = ParameterSpace()
    space1.add(p1)
    space1.add(p2)
    space1.add(p3, lambda p2: p2 > 2.5)
    space1.add(p4, lambda p3: p3 == "bar")

    space_dict = space1.to_dict()
    json_dict = json.dumps(space_dict)

    space2 = ParameterSpace.from_dict(space_dict)
    space3 = ParameterSpace.from_dict(json.loads(json_dict))

    for _ in range(num_samples):
        sample = space1.sample()
        num1 = space1.to_numerical(sample)
        s1 = space1.from_numerical(num1)
        num2 = space2.to_numerical(sample)
        s2 = space2.from_numerical(num2)
        num3 = space3.to_numerical(sample)
        s3 = space3.from_numerical(num2)

        np.testing.assert_array_almost_equal(num1, num2)
        np.testing.assert_array_almost_equal(num1, num3)
        np.testing.assert_array_almost_equal(
            space1.log_likelihood_numerical(num1), space2.log_likelihood_numerical(num2)
        )
        np.testing.assert_array_almost_equal(
            space1.log_likelihood_numerical(num1), space3.log_likelihood_numerical(num3)
        )

        for k, s in sample.items():
            assert s == s1[k] or math.isclose(s, s1[k])
            assert s == s2[k] or math.isclose(s, s2[k])
            assert s == s3[k] or math.isclose(s, s3[k])

    assert space1 == space2


def test_copy():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter(
        "p2", (1e-3, 1e3), prior=TruncatedNormal(0.5, 1), transformation="log"
    )
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space = ParameterSpace()
    space.add(p1)
    space.add(p2)
    space.add(p3, lambda p2: p2 > 2.5)
    space.add(p4, lambda p3: p3 == "bar")

    space_copy = space.copy()
    assert space_copy == space

    copy_sample = space_copy.sample()
    space_sample = space.sample()
    assert copy_sample == space_sample

    space_copy.seed()
    copy_sample = space_copy.sample()
    space_sample = space.sample()
    assert copy_sample != space_sample


def test_dill(num_samples=128):
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter(
        "p2", (1e-3, 1e3), prior=TruncatedNormal(0.5, 1), transformation="log"
    )
    p3 = CategoricalParameter("p3", ["foo", "bar"])
    p4 = OrdinalParameter("p4", ["cold", "warm", "hot"])

    space1 = ParameterSpace()
    space1.add(p1)
    space1.add(p2)
    space1.add(p3, lambda p2: p2 > 2.5)
    space1.add(p4, lambda p3: p3 == "bar")

    tmp = dill.dumps(space1)
    space2 = dill.loads(tmp)

    for _ in range(num_samples):
        sample = space1.sample()
        num1 = space1.to_numerical(sample)
        s1 = space1.from_numerical(num1)
        num2 = space2.to_numerical(sample)
        s2 = space2.from_numerical(num2)

        np.testing.assert_array_almost_equal(num1, num2)
        np.testing.assert_array_almost_equal(
            space1.log_likelihood_numerical(num1), space2.log_likelihood_numerical(num2)
        )

        for k, s in sample.items():
            assert s == s1[k] or math.isclose(s, s1[k])
            assert s == s2[k] or math.isclose(s, s2[k])

    assert space1 == space2


def test_fix_fails():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))
    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    with pytest.raises(ValueError):
        space.fix(p3=None)

    with pytest.raises(ValueError):
        space.fix(p1=-10)

    space.fix(p2=3.0)

    with pytest.raises(ValueError):
        space.to_numerical({"p1": 0.0, "p2": 2.0})  # wrong value for p2

    with pytest.raises(ValueError):
        space.to_numerical({"p1": 0.0})  # missing value for p2


def test_fix_no_conditions():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))
    p3 = CategoricalParameter("p3", ["foo", "bar", "baz"])
    space = ParameterSpace()
    for p in [p1, p2, p3]:
        space.add(p)

    assert len(space) == 3

    space.fix(p3="bar")
    assert len(space) == 2

    sample = space.sample()

    assert len(sample) == 3
    assert "p3" in sample

    num_sample_1 = space.to_numerical(sample)
    val_sample = space.from_numerical(num_sample_1)

    assert sample == val_sample
    assert len(num_sample_1) == 2


def test_fix_with_conditions():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))
    p3 = CategoricalParameter("p3", ["foo", "bar", "baz"])
    p4 = CategoricalParameter("p4", ["foo", "bar", "baz"])
    space = ParameterSpace()
    space.add(p1)
    space.add(p2, lambda p1: p1 <= 0)
    space.add(p3, lambda p1: p1 >= 0)
    space.add(p4, lambda p3: p3 == "foo")

    assert len(space) == 4

    for _ in range(16):
        c = space.sample()
        assert "p1" in c
        assert ("p2" in c) == (c["p1"] <= 0)
        assert ("p3" in c) == (c["p1"] >= 0)

        if c.get("p3", "") == "foo":
            assert c["p4"] in ["foo", "bar", "baz"]
        else:
            assert "p4" not in c

    space.fix(p1=1)
    assert len(space) == 2
    assert space["p3"]["condition"].empty()
    assert not space["p4"]["condition"].empty()
    assert space.has_conditions()

    for _ in range(16):
        c = space.sample()
        assert c["p1"] == 1
        assert "p2" not in c

        if c.get("p3", "") == "foo":
            assert c["p4"] in ["foo", "bar", "baz"]
        else:
            assert "p4" not in c

    space.fix(p3="foo")
    assert len(space) == 1
    assert not space.has_conditions()

    for _ in range(16):
        c = space.sample()
        assert c["p1"] == 1
        assert "p2" not in c
        assert c["p3"] == "foo"
        assert c["p4"] in ["foo", "bar", "baz"]


def test_fix_with_inactive_constants():
    """Make sure that inactive parameters passed to `fix` don't end up as constants"""
    space = ps.ParameterSpace()
    space.add(ps.CategoricalParameter("model", ["a", "b"]))
    space.add(
        ps.ContinuousParameter("v1", bounds=[0, 1]),
        condition=lambda model: model == "a",
    )
    space.add(
        ps.ContinuousParameter("v2", bounds=[0, 1]),
        condition=lambda model: model == "b",
    )
    space.fix(model="b", v1=0.5)

    sample = space.sample()

    assert len(space) == 1
    assert len(sample) == 2
    assert sample["model"] == "b"
    assert not space.has_conditions()


def test_fix_parameter_constant_names():
    p1 = IntegerParameter("p1", (-5, 5))
    p2 = ContinuousParameter("p2", (0, 5))

    space = ParameterSpace()

    space.add(p1)
    space.add(p2)

    space.fix(p2=3)
    parameter_names = space.get_parameter_names()
    constant_names = space.get_constant_names()

    assert "p1" in parameter_names
    assert "p2" not in parameter_names
    assert "p2" in constant_names
    assert "p1" not in constant_names


def test_integer_parameter_with_log_transform_on_boundaries():
    """Make sure that the log transformation works on the boundaries of the interval."""
    space = ps.ParameterSpace()
    space.add(ps.IntegerParameter("p1", bounds=(1, 32), transformation="log"))

    config = space.from_numerical(np.array([0.0]))
    vector = space.to_numerical(config)
    assert config["p1"] == 1
    assert (vector >= 0).all()
    assert (vector <= 1).all()

    config = space.from_numerical(np.array([1.0]))
    vector = space.to_numerical(config)
    assert config["p1"] == 32
    assert (vector >= 0).all()
    assert (vector <= 1).all()


def test_continuous_parameter_with_log_transform_on_boundaries():
    """Make sure that the log transformation works on the boundaries of the interval."""
    space = ps.ParameterSpace()
    space.add(ps.ContinuousParameter("p1", bounds=(1.0, 32.0), transformation="log"))

    config = space.from_numerical(np.array([0.0]))
    vector = space.to_numerical(config)
    assert config["p1"] == 1.0
    assert (vector == 0).all()

    config = space.from_numerical(np.array([1.0]))
    vector = space.to_numerical(config)
    assert config["p1"] == 32.0
    assert (vector == 1).all()


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
