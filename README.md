# ANN Tester
This module runs several tests on a multilayer perceptron (MLP) regression fit.  MLP's have been theoretically proven able to reproduce continuous functions within an arbitrarily small epsilon.  These tests are designed to empirically study the amount of training data required to reproduce various functions and the interaction with the number of "neurons" and layers in the MLP.  Also of interest is the extensibility of ANN models for extrapolation or interpolation of "voids" in the training data.

## Installation
```
conda create -n <env> --file requirements.txt
conda activate <env>
python ann_tests/run_tests.py
```

## Test options
Three tests are available.  'basic' is a test of three ANN architectures with different amounts of training data.  'extrap' is a test of input values outside of the training data range.  'interp' is a test of input values within a subset of the training data range which was excluded from training. 'dim' is a test of increasing dimensionality of the input features.

```
usage: run_tests.py [-h] [--test {interp,extrap,basic,dim,all}]

Run ANN regression tests

optional arguments:
  -h, --help            show this help message and exit
  --test {interp,extrap,basic,dim,all}
                        Test mode
```