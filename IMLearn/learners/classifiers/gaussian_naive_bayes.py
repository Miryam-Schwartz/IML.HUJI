from typing import NoReturn

from numpy.linalg import inv, det

from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.array([np.sum(y == k) for k in self.classes_]) / y.shape[0]
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for i in range(self.classes_.shape[0]):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)

        self.vars_ = np.zeros(self.mu_.shape)
        for i in range(self.classes_.shape[0]):
            inner = X[y == self.classes_[i]] - self.mu_[i]
            self.vars_[i] = np.sum((inner * inner), axis=0) / (sum(y == self.classes_[i]) - 1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        return self.classes_[np.argmax(likelihood, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        output = np.zeros((X.shape[0], self.classes_.shape[0]))
        d_features = X.shape[1]
        for i in range(self.classes_.shape[0]):
            cov = np.diag(self.vars_[i])
            mult = np.sum(((X - self.mu_[i]) @ inv(cov) * (X - self.mu_[i])), axis=1)
            output[:, i] = (1 / np.sqrt(np.power(2 * np.pi, d_features) * det(cov))) * \
                           np.exp(-0.5 * mult) * self.pi_[i]
        return output

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
