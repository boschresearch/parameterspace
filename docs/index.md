# Overview

The package **parameterspace** is used to define parameter spaces consisting of mixed
types (continuous, integer, categorical) with conditionalities and priors.
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

The package can be installed from [PyPi](https://pypi.org/project/parameterspace/) with:

```bash
pip install parameterspace
```

## ConfigSpace Compatibility

In case you are used to working with
[ConfigSpace](https://github.com/automl/ConfigSpace/) or for other reasons have space
definitions in the `ConfigSpace` format around, you can convert them into
`ParameterSpace` instances with ease.
Just note that any colons `:` in the `ConfigSpace` parameter names will be converted to
underscores `_`.

```python
import json
from parameterspace.configspace_utils import parameterspace_from_configspace_dict

with open("config_space.json", "r") as fh:
  cs = json.load(fh)

ps = parameterspace_from_configspace_dict(cs)
```
