from sklearn.neural_network import MLPRegressor
import numpy as np

def get():
    return [
        MLPRegressor(hidden_layer_sizes=(5, 5),
                     learning_rate_init=0.01,
                     early_stopping=True),

        MLPRegressor(hidden_layer_sizes=(50, 50),
                     learning_rate_init=0.01,
                     early_stopping=True),

        MLPRegressor(hidden_layer_sizes=np.full(25, 5),
                     learning_rate_init=0.01,
                     early_stopping=True)
    ]
