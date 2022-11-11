# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import math

import numpy as np
import pytest

from parameterspace import utils


def test_extract_lambda_information():
    """Test parsing of some basic functions.

    Sorry for the mess, but using pytest.mark.parametrize does not work here, because
    the parsing assumes the lambda function to be defined in a line that typically looks
    like this: ps.add(pramater, lambda p1, p2: f(p1, p2))
    Due to the naive implementation, the parser fails when these lambdas are defined in
    a list, which would be the case with the parametrize feature of pytest.
    """
    functions = []
    expected_variables = []
    expected_bodies = []

    functions.append(lambda p1, p2: p1**2 + p2**2 < 1)
    expected_variables.append(["p1", "p2"])
    expected_bodies.append("p1**2 + p2**2 < 1")

    functions.append(lambda p1, p2: math.sin(p1 + p2))
    expected_variables.append(["p1", "p2"])
    expected_bodies.append("math.sin(p1 + p2)")

    for lambda_fn, expected_vars, expected_body in zip(
        functions, expected_variables, expected_bodies
    ):
        variables, body = utils.extract_lambda_information(
            inspect.getsourcelines(lambda_fn)[0]
        )

        assert variables == expected_vars
        assert body == expected_body


@pytest.mark.parametrize(
    "label,function,expected_to_pass",
    [
        ("squared", (lambda p1, p2: p1**2 + p2**2 < 1), True),
        ("math", (lambda p1, p2: math.sin(p1 + p2)), True),
        # string expressions should be ok
        ("string", (lambda p1: p1 == "foo"), True),
        # tuple member check should be ok
        ("tuple member", (lambda p1: p1 in ("a", "b")), True),
        # logic operators should work too
        ("logic", (lambda p1, p2: p1 == 1 or not p2 == 1 and p1 != p2), True),
        # Uneven single quotes should work
        ("uneven single", (lambda p1: p1 == "a single ' is okay"), True),
        # Uneven double quotes should work
        ("uneven double", (lambda p1: p1 == 'a single " is okay'), True),
        # eval is a red flag
        # pylint: disable=eval-used
        ("eval", (lambda: eval("print('I could be malicious!')")), False),
        # body can't be too long
        # pylint: disable=line-too-long
        (
            "long body",
            lambda: (
                "Very long string to trigger the upper character limit. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores"
            ),
            False,
        ),
        # backslash is a forbidden character
        ("backslash", (lambda: print("\\")), False),
        # numpy is currently not allowed
        # pylint: disable=unnecessary-lambda
        ("numpy", (lambda p1: np.cos(p1)), False),
    ],
)
def test_verify_lambda(label, function, expected_to_pass):
    """Test verification of some example lambda functions

    See comment in `test_extract_lambda_information` above.
    The `label` argument is mainly intended to improve readability of sub-test names.
    """

    variables, body = utils.extract_lambda_information(
        inspect.getsourcelines(function)[0]
    )
    assert expected_to_pass == utils.verify_lambda(
        variables, body
    ), f"{label} expected {expected_to_pass}"


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
