import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_loss, test_loss = (np.zeros((n_learners,)), np.zeros((n_learners,)))
    for i in range(1, n_learners + 1):
        train_loss[i - 1] = adaboost.partial_loss(train_X, train_y, i)
        test_loss[i - 1] = adaboost.partial_loss(test_X, test_y, i)

    fig = go.Figure()

    axis_x = list(range(1, n_learners + 1))

    fig.add_trace(go.Scatter(x=axis_x, y=train_loss, name="train errors"))

    fig.add_trace(go.Scatter(x=axis_x, y=test_loss, name="test errors"))

    fig.update_layout(
        title="Train and Test Errors as Function of Number of Fitted Learners",
        xaxis_title="numbers of fitted learners",
        yaxis_title="error",
        legend_title="data sets:",
    )

    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    test_y_corrected = np.where(test_y < 0, 0, 1)

    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"Iteration: {t}" for t in T])

    for i, t in enumerate(T):
        def temp_predict(X: np.ndarray) -> np.ndarray:
            return adaboost.partial_predict(X, t)

        fig.add_traces([decision_surface(temp_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y_corrected, symbol=symbols[test_y_corrected],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=1, cols=i + 1)

    fig.update_layout(title="Decision Boundary Obtained by Using the Ensemble Up To Specified Size",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    size_min_err = np.argmin(test_loss) + 1
    accuracy = 1 - test_loss[size_min_err - 1]

    def temp_predict(X: np.ndarray) -> np.ndarray:
        return adaboost.partial_predict(X, size_min_err)

    fig = go.Figure()
    fig.add_traces([decision_surface(temp_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y_corrected, symbol=symbols[test_y_corrected],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout(title=f"Decision Boundary of Ensemble of Size {size_min_err}. Accuracy: {accuracy}",
                      width=750, height=750)
    fig.show()

    # Question 4: Decision surface with weighted samples
    train_y_corrected = np.where(train_y < 0, 0, 1)
    D = adaboost.D_ / np.max(adaboost.D_) * 10
    fig = go.Figure()
    fig.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=D, color=train_y_corrected, symbol=symbols[train_y_corrected],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout(title="Training set with point size proportional to it’s weight", width=750, height=750)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(noise=0.4)
