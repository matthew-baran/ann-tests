import estimators

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time

X = np.linspace(-10, 10, 10000).reshape(-1, 1)
y = np.ravel(np.square(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)

estimator_list = estimators.get()

figure = plt.figure()

for idx, est in enumerate(estimator_list):
    print("Training MLPRegressor...")
    tic = time()

    est.fit(X_train, y_train)
    print("done in {:.3f}s".format(time() - tic))
    print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))
    

    y_est = est.predict(X_test)
    print("Mean abs error: {:.2f}".format(np.mean(np.absolute(y_est-y_test))))

    plt.subplot(1, 3, idx+1)
    plt.plot(X, y)
    plt.scatter(X_test, y_est, color='r')

    y_est = est.predict(X_train)

    plt.scatter(X_train, y_est, color='g')

plt.show()


