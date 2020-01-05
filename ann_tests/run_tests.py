import estimators

import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from time import time


def annStructureTest():
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

        ax_list[sample_idx, 0].set_ylabel(
            str(len(X_train)) + " training samples")

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

            ax_list[sample_idx, estimator_idx].scatter(
                X_test, y_est, color='r')
            ax_list[sample_idx, estimator_idx].plot(X, y)
            ax_list[sample_idx, estimator_idx].set_title(
                descriptions[estimator_idx])
            ax_list[sample_idx, estimator_idx].set_xlabel(
                "$\epsilon_\mu=${:.2f} $\epsilon_{{max}}=${:.2f}".format(np.mean(err), np.max(err)))

    plt.tight_layout()


def extrapolationTest():
    num_samples = 1e5
    est = MLPRegressor(hidden_layer_sizes=(50, 50),
                       learning_rate_init=0.01,
                       early_stopping=True)

    plt.figure()
    plt.title(
        "ANN regression fit of $y=x^2$, with extrapolation outside of [-10 10]")

    X_train = np.linspace(-10, 10, num_samples).reshape(-1, 1)
    y_train = np.ravel(np.square(X_train))

    X_test = np.linspace(-100, 100, 10e3).reshape(-1, 1)
    y_test = np.ravel(np.square(X_test))

    plt.ylabel(str(len(X_train)) + " training samples")

    print("Training with " + str(len(X_train)) +
          " samples and 50 neurons, 2 layers...")

    tic = time()
    est.fit(X_train, y_train)
    print("done in {:.3f}s".format(time() - tic))

    y_est = est.predict(X_test)

    err = np.absolute(y_est - y_test)
    rel_err = np.absolute(np.divide(y_est - y_test, y_test))

    print("Mean error: {:.2f}".format(np.mean(err)))
    print("Max error: {:.2f}".format(np.max(err)))
    print("Mean relative error: {:.2f}\n".format(np.mean(rel_err)))

    plt.scatter(
        X_test, y_est, color='r')
    plt.plot(X_test, y_test)
    plt.xlabel(
        "$\epsilon_\mu=${:.2f} $\epsilon_{{max}}=${:.2f}".format(np.mean(err), np.max(err)))

    plt.tight_layout()


def interpolationTest():
    num_samples = 1e5
    est = MLPRegressor(hidden_layer_sizes=(50, 50),
                       learning_rate_init=0.01,
                       early_stopping=True)

    plt.figure()
    plt.title(
        "ANN regression fit of $y=x^2$, with void interpolation in [-10, 10]")

    X_train = np.linspace(-100, 100, num_samples)
    X_train = X_train[np.where(np.abs(X_train) > 10)].reshape(-1, 1)
    y_train = np.ravel(np.square(X_train))

    X_test = np.linspace(-100, 100, 10e3).reshape(-1, 1)
    y_test = np.ravel(np.square(X_test))

    plt.ylabel(str(len(X_train)) + " training samples")

    print("Training with " + str(len(X_train)) +
          " samples and 50 neurons, 2 layers...")

    tic = time()
    est.fit(X_train, y_train)
    print("done in {:.3f}s".format(time() - tic))

    y_est = est.predict(X_test)

    err = np.absolute(y_est - y_test)
    rel_err = np.absolute(np.divide(y_est - y_test, y_test))

    print("Mean error: {:.2f}".format(np.mean(err)))
    print("Max error: {:.2f}".format(np.max(err)))
    print("Mean relative error: {:.2f}\n".format(np.mean(rel_err)))

    plt.scatter(
        X_test, y_est, color='r')
    plt.plot(X_test, y_test)
    plt.xlabel(
        "$\epsilon_\mu=${:.2f} $\epsilon_{{max}}=${:.2f}".format(np.mean(err), np.max(err)))
    plt.xlim((-20, 20))
    plt.ylim((min(-5, np.min(y_est)-5), 405))

    plt.tight_layout()

def dimensionTest():
    num_samples = 1e5
    est = MLPRegressor(hidden_layer_sizes=(50, 50),
                       learning_rate_init=0.01,
                       early_stopping=True)

    num_dims = [1,3,5]

    figure, ax_list = plt.subplots(len(num_dims), 1, sharex=True, sharey=True)
    figure.suptitle(r'ANN regression fit of $y=\Vert x \Vert ^2, x \in \mathbb{{R}}^n$')

    for idx, n in enumerate(num_dims):
        X = np.random.rand(int(num_samples), int(n))*20 - 10
        y = np.sum(np.square(X), 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            random_state=0)

        ax_list[idx].set_ylabel(
            str(len(X_train)) + " training samples")

        print("Training with " + str(len(X_train)) +
                " samples and 50 neurons, 2 layers...")

        tic = time()
        est.fit(X_train, y_train)
        print("done in {:.3f}s".format(time() - tic))

        y_est = est.predict(X_test)

        err = np.absolute(y_est - y_test)
        rel_err = np.absolute(np.divide(y_est - y_test, y_test))

        print("Mean error: {:.2f}".format(np.mean(err)))
        print("Max error: {:.2f}".format(np.max(err)))
        print("Mean relative error: {:.2f}\n".format(np.mean(rel_err)))

        X_test = np.linspace(-10, 10, 1e3).reshape(-1, 1)
        X_test = np.concatenate((X_test, np.zeros((int(1e3),int(n-1)))), axis=1)
        y_test = np.ravel(np.square(X_test[:,0]))

        y_est = est.predict(X_test)

        ax_list[idx].scatter(X_test[:,0], y_est, color='r')
        ax_list[idx].plot(X_test[:,0], y_test)
        ax_list[idx].set_title("Dimension 1 of {:d}".format(n))
        ax_list[idx].set_xlabel(
            "$\epsilon_\mu=${:.2f} $\epsilon_{{max}}=${:.2f}".format(np.mean(err), np.max(err)))

    plt.tight_layout()


def allTests():
    annStructureTest()
    extrapolationTest()
    interpolationTest()
    dimensionTest()


def notImplemented():
    print("Test mode not yet implemented.")


def main(test_mode):
    modes = {
        "basic": annStructureTest,
        "extrap": extrapolationTest,
        "interp": interpolationTest,
        "dim": dimensionTest,
        "all": allTests
    }

    modes[test_mode]()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANN regression tests")
    parser.add_argument("--test", type=str,
                        choices={"basic", "extrap", "interp", "dim", "all"}, default="basic",
                        help="Test mode")
    args = parser.parse_args()

    main(args.test)
