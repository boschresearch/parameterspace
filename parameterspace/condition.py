# Copyright (c) 2021 - for information on the respective copyright owner see the
# NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Callable

from parameterspace import utils


class Condition:
    """Class that handels the logic of conditioning parameters on others in a
    ParameterSpace."""

    def __init__(self, lambda_fn: Callable = None):
        """
        Args:
            lambda_fn: A callable whose signature contains valid names of other
                parameters. It returns True if the associate parameter is active,
                and False otherwise.
        """
        self.functions = []
        self.function_texts = []
        self.varnames = []
        self.all_varnames = set()

        if lambda_fn is not None:
            self.functions = [lambda_fn]
            varnames, function_text = utils.extract_lambda_information(
                inspect.getsourcelines(lambda_fn)[0]
            )
            # self.varnames = [lambda_fn.__code__.co_varnames]
            self.varnames = [varnames]
            self.function_texts = [function_text]
            self.all_varnames = set(lambda_fn.__code__.co_varnames)

    def __repr__(self):
        if not self.all_varnames:
            return "No conditions!"
        string = "Condition(s) depend(s) on {}.\n".format(
            self.all_varnames
        ) + "\n".join(self.function_texts)
        return string

    def __call__(self, config=None):
        if not self.all_varnames:
            return True
        config = {} if config is None else config
        for (vn, fn) in zip(self.varnames, self.functions):
            if not set(vn).issubset(config.keys()):
                raise ValueError("Not all variables set to evaluate the condition!")
            args = [config[n] for n in vn]
            if None in args:
                return False
            if not fn(*args):
                return False
        return True

    def empty(self) -> bool:
        """Check if this Condition is not trivial.

        Returns:
            [description]
        """
        return not bool(self.all_varnames)

    def merge(self, other: Condition = None) -> Condition:
        """Concatenates two Conditions to allow for hierarchical spaces.

        Args:
            other: [description]

        Returns:
            [description]
        """
        if other is not None:
            self.all_varnames = self.all_varnames.union(other.all_varnames)
            self.functions.extend(other.functions)
            self.function_texts.extend(other.function_texts)
            self.varnames.extend(other.varnames)
        return self

    def to_dict(self) -> dict:
        """[summary]

        Returns:
            [description]
        """
        return_dict = {}
        for i, (varnames, function_text) in enumerate(
            zip(self.varnames, self.function_texts)
        ):
            return_dict[str(i)] = {"varnames": varnames, "function_text": function_text}
        return return_dict

    @staticmethod
    def from_dict(
        dict_representation: dict, verify_lambda: Callable = utils.verify_lambda
    ) -> Condition:
        """[summary]

        Args:
            dict_representation: [description]
            verify_lambda: [description]

        Raises:
            RuntimeError: If "function_text" in in `dict_representation` is
                not considered save.

        Returns:
            [description]
        """

        return_condition = Condition()

        for k in range(len(dict_representation)):
            varnames = dict_representation[str(k)]["varnames"]
            function_text = dict_representation[str(k)]["function_text"]

            if not verify_lambda(varnames, function_text):
                raise RuntimeError(f"Source {function_text} is not considered save!")

            return_condition.function_texts.append(function_text)
            return_condition.varnames.append(varnames)
            return_condition.all_varnames |= set(varnames)

            # pylint: disable=eval-used
            return_condition.functions.append(
                eval(f"lambda {', '.join(varnames)}: {function_text}")
            )
            # pylint: enable=eval-used

        return return_condition

    def __eq__(self, other):
        if self.all_varnames != other.all_varnames:
            return False
        if len(self.function_texts) != len(other.function_texts):
            return False

        for t1, t2 in zip(self.function_texts, other.function_texts):
            if t1 != t2:
                return False

        return True
