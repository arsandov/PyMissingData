PyMissingData
================

We use Bayesian networks and a probabilistic graphical model that allows us to perform inference to predict missing values given
observed data and dependency relationships between variables. To learn a Bayesian network from incomplete
data, we use an iterative algorithm that utilizes sampling methods and expectation maximization to estimate the
distributions and probabilistic dependencies of variables from data with missing values.

Installation
------------

The library can be installed through pip as:
 
```linux
pip install PyMissingData
```

Also keep in mind that this library has the following dependencies:
- pandas
- numpy
- scipy
- libpgm (included because we use a modified version)

