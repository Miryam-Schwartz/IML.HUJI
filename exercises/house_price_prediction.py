from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df[df['id'] != 0]
    df = df[df['price'] > 0]
    df = df[df['bedrooms'] >= 0]
    df = df[df['bathrooms'] >= 0]
    df = df[df['sqft_living'] > 0]
    df = df[df['sqft_above'] > 0]
    df = df[df['sqft_basement'] >= 0]
    df = df[df['yr_built'] > 0]
    df = df[df['yr_renovated'] >= 0]
    df = df[df['sqft_living15'] > 0]
    df = df[df['sqft_lot15'] > 0]
    del df['id']
    del df['sqft_living']  # depend on other colmuns: 'sqft_above' and 'sqft_basement'

    # date
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    del df['date']
    df = pd.get_dummies(df, columns=['month'])

    # year built / year renovated
    df['max_built/renovated_yr'] = df[['yr_built', 'yr_renovated']].max(axis=1)
    df = df.rename(columns={'yr_renovated': 'is_renovated'})
    df['is_renovated'] = (df['is_renovated'] != 0).astype(int)
    del df['yr_built']

    # zipcode
    df = pd.get_dummies(df, columns=['zipcode'])

    label = df.pop('price')
    return df, label


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        pearson_correlation = np.cov(X[col], y) / np.std(X[col]) * np.std(y)
        fig = go.Figure([go.Scatter(x=X[col], y=y, mode='markers')],
                        xaxis_title="$x\\text{ - feature values}$",
                        yaxis_title="$y\\text{ - respones}$")

        fig.update_layout(title_text= 'Feature Name:' + col)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data("C:\\Users\\Miryam\\Documents\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y)

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
