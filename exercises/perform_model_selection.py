from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = (X+3)*(X+2)*(X+1)*(X-1)*(X-2)
    y_noised = y_noiseless + np.random.normal(0, noise, n_samples)
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y_noised), 2/3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_noiseless, name='true model'))
    fig.add_trace(go.Scatter(x=train_x.to_numpy()[:, 0], y=train_y.to_numpy()[:, 0], mode='markers', name='train sample'))
    fig.add_trace(go.Scatter(x=test_x.to_numpy()[:, 0], y=test_y.to_numpy()[:, 0], mode='markers', name='test sample'))
    fig.update_layout(title=f"Model of polynom y=(X+3)*(X+2)*(X+1)*(X-1)*(X-2). Num samples: {n_samples}. Noise: {noise}.",
                      xaxis_title='X', yaxis_title='f(x)', legend_title='data set:')
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    error_train_lst = []
    error_validate_lst = []
    for k in range(11):
        estimator = PolynomialFitting(k)
        error_train, error_validate = cross_validate(estimator, train_x.to_numpy()[:, 0], train_y.to_numpy()[:, 0],
                                                     mean_square_error, 5)
        error_train_lst.append(error_train)
        error_validate_lst.append(error_validate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(11)), y=error_train_lst, mode='markers', name='train error'))
    fig.add_trace(go.Scatter(x=list(range(11)), y=error_validate_lst, mode='markers', name='validation error'))
    fig.update_layout(title=f"Train and validation error as function of polynom degree. Num samples: {n_samples}. Noise: {noise}.",
                      xaxis_title='K - polynom degree',
                      yaxis_title='error', legend_title='data set:')
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(error_validate_lst)
    validation_error_of_k = error_validate_lst[k]
    estimator = PolynomialFitting(k)
    estimator.fit(train_x.to_numpy()[:, 0], train_y.to_numpy()[:, 0])
    test_error = estimator.loss(test_x.to_numpy()[:, 0], test_y.to_numpy()[:, 0])
    print(f"Num samples: {n_samples}. Noise: {noise}. K* is {k}. Validation error: {validation_error_of_k.__round__(2)}. test error:  {test_error.__round__(2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=50)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    possible_lams = np.linspace(0, 2, n_evaluations)
    possible_lams = np.delete(possible_lams, 0)
    train_error_lasso = []
    validation_error_lasso = []
    for lam in possible_lams:
        estimator = Lasso(lam, max_iter=500)
        train_error, validate_error = cross_validate(estimator, train_x, train_y, mean_square_error, 5)
        train_error_lasso.append(train_error)
        validation_error_lasso.append(validate_error)

    train_error_ridge = []
    validation_error_ridge = []
    for lam in possible_lams:
        estimator = RidgeRegression(lam)
        train_error, validate_error = cross_validate(estimator, train_x, train_y,
                                                     mean_square_error, 5)
        train_error_ridge.append(train_error)
        validation_error_ridge.append(validate_error)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=possible_lams, y=train_error_ridge, name='Ridge_train error'))
    fig.add_trace(go.Scatter(x=possible_lams, y=validation_error_ridge, name='Ridge_validation error'))
    fig.add_trace(go.Scatter(x=possible_lams, y=train_error_lasso, name='Lasso train error'))
    fig.add_trace(go.Scatter(x=possible_lams, y=validation_error_lasso, name='Lasso validation error'))
    fig.update_layout(title='Train error and validation error for ridge and Lasso algorithms',
                      xaxis_title='lambda - regularization parameter', yaxis_title='error')
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = possible_lams[np.argmin(validation_error_ridge)]
    best_lasso = possible_lams[np.argmin(validation_error_lasso)]

    print(f"best regularization parameter for ridge: {best_ridge}")
    print(f"best regularization parameter for lasso: {best_lasso}")

    ridge_est = RidgeRegression(best_ridge)
    lasso_est = Lasso(best_lasso)
    lse_est = LinearRegression()

    ridge_est.fit(train_x, train_y)
    lasso_est.fit(train_x, train_y)
    lse_est.fit(train_x, train_y)

    print("\nTest Erros:")
    print(f"Ridge regression: {ridge_est.loss(test_x, test_y)}")
    print(f"Lasso regression: {mean_square_error(lasso_est.predict(test_x), test_y)}")
    print(f"Least Squares regression: {lse_est.loss(test_x, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

