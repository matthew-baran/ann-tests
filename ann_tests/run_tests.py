import estimators

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time

# sample_list = [1e4, 1e5, 1e6]
sample_list = [1e3, 1e4, 1e5]
estimator_list, descriptions = estimators.get()

figure, ax_list = plt.subplots(len(sample_list), len(
    estimator_list), sharex=True, sharey=True)
figure.suptitle("ANN regression fit of $y=x^2$")

for sample_idx, num_samples in enumerate(sample_list):
    X = np.linspace(-10, 10, num_samples).reshape(-1, 1)
    y = np.ravel(np.square(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=0)

    ax_list[sample_idx, 0].set_ylabel(str(len(X_train)) + " training samples")

    for estimator_idx, est in enumerate(estimator_list):
        print("Training with " + str(len(X_train)) +
              " samples and " + descriptions[estimator_idx] + "...")

        tic = time()
        est.fit(X_train, y_train)
        print("done in {:.3f}s".format(time() - tic))

        y_est = est.predict(X_test)

        err = np.absolute(y_est - y_test)
        rel_err = np.absolute(np.divide(y_est - y_test, y_test))

        print("Mean error: {:.2f}".format(np.mean(err)))
        print("Max error: {:.2f}".format(np.max(err)))
        print("Mean relative error: {:.2f}\n".format(np.mean(rel_err)))

        ax_list[sample_idx, estimator_idx].scatter(X_test, y_est, color='r')
        ax_list[sample_idx, estimator_idx].plot(X, y)
        ax_list[sample_idx, estimator_idx].set_title(
            descriptions[estimator_idx])
        ax_list[sample_idx, estimator_idx].set_xlabel(
            "$\epsilon_\mu=${:.2f} $\epsilon_{{max}}=${:.2f}".format(np.mean(err), np.max(err)))


plt.tight_layout()
plt.show()
