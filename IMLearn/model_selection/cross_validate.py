from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m_samples = y.shape[0]
    leng_of_part = m_samples // cv
    sum_error_validate = 0
    sum_error_train = 0
    for i in range(cv):
        indexes_to_delete = list(range(i * leng_of_part, (i + 1) * leng_of_part))
        X_temp, y_temp = np.delete(arr=X, obj=indexes_to_delete, axis=0), \
                         np.delete(y, indexes_to_delete, axis=0)
        estimator.fit(X_temp, y_temp)
        predicted_validate = estimator.predict(X[indexes_to_delete])
        predicted_train = estimator.predict(X_temp)
        sum_error_validate += scoring(y[indexes_to_delete], predicted_validate)
        sum_error_train += scoring(y_temp, predicted_train)
    return sum_error_train / cv, sum_error_validate / cv
