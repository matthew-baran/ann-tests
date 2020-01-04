# ANN Tester
This module runs several tests on a multilayer perceptron (MLP) regression fit.  MLP's have been theoretically proven able to reproduce continuous functions within an arbitrarily small epsilon.  These tests are designed to empirically study the amount of training data required to reproduce various functions and the interaction with the number of "neurons" and layers in the MLP.  Also of interest is the extensibility of ANN models for extrapolation or interpolation of "voids" in the training data.

## Installation
```
pip install -r requirements.txt
python ann_tests/run_tests.py
```