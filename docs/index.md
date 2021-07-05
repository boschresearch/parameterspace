# Overview

The package **parameterspace** is used to define parameter spaces consisting of mixed
types (continuous, integers, catergoricals) with conditionalities and priors.
It allows for easy specification of the parameters and their dependencies.

The [`ParameterSpace`](parameterspace/parameterspace) object can then be used to sample
random configurations from the prior and convert any valid configuration into a
numerical representation. This numerical representation has the following properties:

- it results in a numpy `ndarray` of type `numpy.float64`
- transformed representation between 0 and 1 (uniform) including integers, ordinal and
  categorical parameters
- inactive parameters are masked as `numpy.nan` values

This allows to easily use optimizers that expect continuous domains to be used on more
complicated problems because [`ParameterSpace`](parameterspace/parameterspace) can
convert any numerical vector representation inside the unit hypercube into a valid
configuration.
The function might not be smooth, but for robust methods (like genetic
algorithms/evolutionary strategies) this might still be valuable.

The package can be installed from the Bosch internal PyPi mirrors with:

```bash
pip install parameterspace
```

## License

Copyright (c) 2021 - for information on the respective copyright owner
see the NOTICE file and/or the repository https://github.com/boschresearch/parameterspace

SPDX-License-Identifier: Apache-2.0
