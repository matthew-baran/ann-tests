import numpy as np
from sklearn.neural_network import MLPRegressor


def get():
    estimator_list = [
        MLPRegressor(hidden_layer_sizes=(5, 5), learning_rate_init=0.01, early_stopping=True),
        MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, early_stopping=True),
        MLPRegressor(
            hidden_layer_sizes=np.full(20, 5), learning_rate_init=0.01, early_stopping=True
        ),
    ]

    descriptions = ["5 neurons, 2 layers", "50 neurons, 2 layers", "5 neurons, 20 layers"]

    return estimator_list, descriptions
