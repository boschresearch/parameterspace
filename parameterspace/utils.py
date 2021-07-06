# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from functools import wraps
from typing import Callable, Iterable, Tuple

import numpy as np


def store_init_arguments(init_method: Callable) -> Callable:
    """Stores init arguments (including keyword arguments) and auto converts
    `numpy.ndarray`s to `list`s for json serializability.

    Args:
        init_method: [description]

    Returns:
        [description]
    """

    @wraps(init_method)
    def wrapper(self, *args, **kwargs):
        args = [a.tolist() if isinstance(a, np.ndarray) else a for a in args]

        for k, k_value in kwargs.items():
            if isinstance(k_value, np.ndarray):
                kwargs[k] = k_value.tolist()

        self._init_args = args
        self._init_kwargs = kwargs
        init_method(self, *args, **kwargs)

    return wrapper


def extract_lambda_information(source_lines: Iterable) -> Tuple[list, str]:
    """[summary]

    Args:
        source_lines: [description]

    Raises:
        RuntimeError: If function definition is not a valid Lambda function.

    Returns:
        [description]
    """
    condensed_code = "".join(source_lines).replace(os.linesep, "")  # join lines
    condensed_code = " ".join(
        condensed_code.split()
    )  # replace multiple spaces by a single one

    try:
        signature, body = condensed_code.split("lambda")[1].split(":")
    except IndexError as e:
        raise RuntimeError(
            "The function definition does not look like a valid Lambda function:\n"
            + "".join(source_lines)
        ) from e

    while len(body) > 1:
        lambda_def = f"lambda {signature}: {body}"
        try:
            _ = compile(lambda_def, "<unused filename>", "eval")
            break
        except SyntaxError:
            body = body[:-1]

    variables = [s.strip() for s in signature.split(",")]

    return variables, body.strip().rstrip(",")


def verify_lambda(variables: list, body: str) -> bool:
    """Check serialized lambda expression for malicious code.

    Args:
        variables: [description]
        body: [description]

    Returns:
        [description]
    """

    if len(body) > 200:
        return False

    if "eval(" in body:
        return False

    blacklisted_characters = "\\;"

    for c in blacklisted_characters:
        if c in body:
            return False

    white_listed_chars = ".0123456789+-*/()<=!> "
    white_listed_functions = [
        "math.sin",
        "math.cos",
        "math.exp",
        "math.log",
        "or",
        "and",
        "not",
    ]

    for vn in variables:
        body = body.replace(vn, "")

    for fn in white_listed_functions:
        body = body.replace(fn, "")

    for c in white_listed_chars:
        body = body.replace(c, "")

    # remove all single quoted strings
    matches = re.findall(r"\'(.+?)\'", body)
    for m in matches:
        body = body.replace("'" + m + "'", "")

    # remove all double quoted strings
    matches = re.findall(r"\"(.+?)\"", body)
    for m in matches:
        body = body.replace('"' + m + '"', "")

    if len(body) > 0:
        print("body:", body)
        return False

    return True
