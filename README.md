# ParameterSpace

[![Actions Status](https://github.com/boschresearch/parameterspace/workflows/ci-cd-pipeline/badge.svg)](https://github.com/boschresearch/parameterspace/actions)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/parameterspace)](https://pypi.org/project/parameterspace/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/parameterspace)](https://pypi.org/project/parameterspace/)
[![License: Apache-2.0](https://img.shields.io/github/license/boschresearch/parameterspace)](https://github.com/boschresearch/parameterspace/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Contents:**

- [About](#about)
- [Installation](#installation)
- [Development](#development)
  - [Prerequisites](#prerequisites)
  - [Setup environment](#setup-environment)
  - [Running Tests](#running-tests)
  - [Building Documentation](#building-documentation)
- [License](#license)

## About

A package to define parameter spaces consisting of mixed types (continuous, integer,
categorical) with conditions and priors. It allows for easy specification of the
parameters and their dependencies. The `ParameterSpace` object can then be used to
sample random configurations from the prior and convert any valid configuration
into a numerical representation. This numerical representation has the following
properties:

- it results in a Numpy `ndarray` of type `float64`
- transformed representation between 0 and 1 (uniform) including integers, ordinal and
  categorical parameters
- inactive parameters are masked as `numpy.nan` values

This allows to easily use optimizers that expect continuous domains to be used on more
complicated problems because `parameterspace` can convert any numerical vector
representation inside the unit hypercube into a valid configuration. The function might
not be smooth, but for robust methods (like genetic algorithms/evolutionary strategies)
this might still be valuable.

This software is a research prototype. The software is not ready for production use. It
has neither been developed nor tested for a specific use case. However, the license
conditions of the applicable Open Source licenses allow you to adapt the software to
your needs. Before using it in a safety relevant setting, make sure that the software
fulfills your requirements and adjust it according to any applicable safety standards
(e.g. ISO 26262).

## Documentation

**Visit [boschresearch.github.io/parameterspace](https://boschresearch.github.io/parameterspace/)**


## Installation

The `parameterspace` package can be installed from [pypi.org](https://pypi.org):

```
pip install parameterspace
```

## Development

### Prerequisites

- Python >= 3.8
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup environment

To install the package and its dependencies for development run:

```
poetry install
```

Optionally install [pre-commit](https://pre-commit.com) hooks to check code standards
before committing changes:

```
poetry run pre-commit install
```

### Running Tests

The tests are located in the `./tests` folder. The [pytest](https://pytest.org)
framework is used for running them. To run the tests:

```
poetry run pytest ./tests
```

### Building Documentation

To built documentation run from the repository root:

```
poetry run mkdocs build --clean
```

For serving it locally while working on the documentation run:

```
poetry run mkdocs serve
```

## License

`parameterspace` is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in `parameterspace`, see the file
[3rd-party-licenses.txt](3rd-party-licenses.txt).
