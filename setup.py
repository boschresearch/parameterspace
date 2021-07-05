# Copyright (c) 2021 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace
#
# SPDX-License-Identifier: Apache-2.0

from codecs import open
from setuptools import setup, find_packages
from os import path


name = "parameterspace"
version = "0.7"
release = "0.7.10"

description = "Parametrized hierarchical spaces with flexible priors and transformations"
authors = ["Bosch Center for AI, Robert Bosch GmbH"]
url = "https://github.com/boschresearch/blackboxopt"

dependencies = ["numpy>=1.17.0", "scipy>=1.6.0"]
test_dependencies = [
    "pytest>=5.0",
    "pytest-cov",
    "pytest-rerunfailures>=9.0",
    "num2tex",
    "dill",
]
dev_dependencies = ["black==20.8b1", "pylint"] + test_dependencies
example_dependencies = ["notebook", "matplotlib"]
docu_dependencies = ["mkdocs", "mkdocstrings", "mkdocs-material", "mkdocs-jupyter"] + example_dependencies

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=name,
    version=release,
    description=description,
    long_description=long_description,
    url=url,
    author=authors,
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=["License :: OSI Approved :: Apache Software License", "Programming Language :: Python :: 3"],
    license_files=("LICENSE.txt",),
    packages=find_packages(exclude=["doc", "tests"]),
    #  options for documentation builder
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "doc/source"),
            "build_dir": ("setup.py", "doc/build"),
        }
    },
    install_requires=dependencies,
    extras_require={
        "dev": dev_dependencies,
        "test": test_dependencies,
        "examples": example_dependencies,
        "doc": docu_dependencies,
    },
    tests_require=test_dependencies,
)
