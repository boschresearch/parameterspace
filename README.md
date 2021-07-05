# ParameterSpace

* [About](#about)
* [How to use it](#use)
* [How to build and test it](#build)
* [License](#license)

## <a name="about">About</a>

A package to define parameter spaces consisting of mixed types (continuous, integers,
catergoricals) with conditionalities and priors.
It allows for easy specification of the parameters and their dependencies.
The `ParameterSpace` object can then be used to sample random random configurations
from the prior and convert any valid configuration into a numerical representation.
This numerical representation has the following properties:

  - it results in a numpy ndarray of type numpy.float64
  - transformed representation between 0 and 1 (uniform) including integers, ordinal and categorical parameters
  - inactive parameters are masked as  numpy.nan values

This allows to easily use optimizers that expect continuous domains to be used on more
complicated problems because `parameterspace` can convert any numerical vector
representation inside the unit hypercube into a valid configuration.
The function might not be smooth, but for robust methods (like genetic
algorithms/evolutionary strategies) this might still be valuable.

This software is a research prototype.
The software is not ready for production use.
It has neither been developed nor tested for a specific use case.
However, the license conditions of the applicable Open Source licenses allow you to
adapt the software to your needs.
Before using it in a safety relevant setting, make sure that the software fulfills your
requirements and adjust it according to any applicable safety standards
(e.g. ISO 26262).

## <a name="use">How to use it</a>

The parameterspace package can be installed with the Python package manager:

```
pip install parameterspace
```


## <a name="build">How to build and test it</a>


### Getting Started

To install the package and its dependencies for development run:
```
pip install -e .[dev]
```


### Running Tests

The tests are located in the `./tests` folder.
The [Pytest](https://pytest.org) framework is used for running them.
To run the tests:
```
pip install -e .[dev,test]
pytest ./tests
```


### Building Documentation

To build the documentation, one needs to install
parameterspace with doc dependencies:
```
pip install -e .[dev,doc]
```

To built documentiation run from the repository root:
```
mkdocs build --clean
```

For serving it locally while working on the documentation run:
```
mkdocs serve
```

## License

`parameterspace` is open-sourced under the Apache-2.0 license. See the [LICENSE](LICENSE)
file for details.

For a list of other open source components included in `parameterspace`, see the file
[3rd-party-licenses.txt](3rd-party-licenses.txt).
