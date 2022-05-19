# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import math

import pytest

import parameterspace.utils as utils


def test_extract_lambda_information():
    """Test parsing of some basic functions.

    Sorry for the mess, but using pytest.mark.parametrize does not work here, because the parsing
    assumes the lambda function to be defined in a line that typically looks like this:
    ps.add(pramater, lambda p1, p2: f(p1, p2))
    Due to the naive implementation, the parser fails when these lambdas are defined in a list, which would be the case
    with the parametrize feature of pytest.
    """
    functions = []
    expected_variables = []
    expected_bodies = []

    functions.append(lambda p1, p2: p1**2 + p2**2 < 1)
    expected_variables.append(["p1", "p2"])
    expected_bodies.append("p1 ** 2 + p2 ** 2 < 1")

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


def test_verify_lambda():
    """Test verification of some example lambda functions

    See comment in `test_extract_lambda_information` above.
    """

    functions = []
    expected_to_pass = []

    functions.append(lambda p1, p2: p1**2 + p2**2 < 1)
    expected_to_pass.append(True)

    functions.append(lambda p1, p2: math.sin(p1 + p2))
    expected_to_pass.append(True)

    # string expressions should be ok
    functions.append(lambda p1, p2: p1 == "foo")
    expected_to_pass.append(True)

    # logic operators should work too
    functions.append(lambda p1, p2: p1 == 1 or not p2 == 1 and p1 != p2)
    expected_to_pass.append(True)

    # Uneven single quotes should work
    functions.append(lambda p1: p1 == "a single ' is okay")
    expected_to_pass.append(True)

    # Uneven double quotes should work
    functions.append(lambda p1: p1 == 'a single " is okay')
    expected_to_pass.append(True)

    # eval is a red flag
    functions.append(lambda: eval("print('I could be malicious!')"))
    expected_to_pass.append(False)

    # body can't be too long
    functions.append(
        lambda: "Very long string to make trigger the upper character limit. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores "
    )
    expected_to_pass.append(False)

    # backslash is a blacklisted character
    functions.append(lambda: print("\\"))
    expected_to_pass.append(False)

    # numpy is currently not white listed
    functions.append(lambda p1: np.cos(p1))
    expected_to_pass.append(False)

    for lambda_fn, pass_expected in zip(functions, expected_to_pass):
        variables, body = utils.extract_lambda_information(
            inspect.getsourcelines(lambda_fn)[0]
        )
        assert pass_expected == utils.verify_lambda(variables, body)


if __name__ == "__main__":
    pytest.main(["--pdb", "-s", __file__])
