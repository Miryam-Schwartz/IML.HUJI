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
    df.dropna(inplace=True)
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
    del df['lat']
    del df['long']

    # date
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    del df['date']
    df = pd.get_dummies(df, columns=['month'])

    # year built / year renovated
    df['max_built_or_renovated_yr'] = df[['yr_built', 'yr_renovated']].max(axis=1)
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
        pearson_correlation = np.cov(X[col], y)[0, 1] / (np.std(X[col]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[col], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {col} Values and Respone <br>Pearson Correlation "
                               f"{pearson_correlation}",
                         labels={"x": f"{col} Values", "y": "Respone Values"})
        fig.write_image("pearson.correlations.%s.png" % col)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data("C:\\Users\\Miryam\\Documents\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(x, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    linear_regression_estimator = LinearRegression()
    x = list(range(10, 101))
    mean_loss_arr, std_loss_arr = np.zeros((91,)), np.zeros((91,))
    for p in x:
        loss_arr = np.zeros((10,))
        for i in range(10):
            cur_sample, cur_response, temp0, temp1 = split_train_test(train_x, train_y, (p / 100))
            linear_regression_estimator.fit(cur_sample.to_numpy(), cur_response.to_numpy())
            loss_arr[i] = linear_regression_estimator.loss(test_x.to_numpy(), test_y.to_numpy())
        # mean_loss, std_loss = df_loss.mean(), df_loss.std()
        mean_loss_arr[p-10] = np.mean(loss_arr)
        std_loss_arr[p - 10] = np.std(loss_arr)
        # mean_loss_arr.append(mean_loss)
        # std_loss_arr.append(std_loss)
    # mean_loss_arr = pd.DataFrame(mean_loss_arr)
    # std_loss_arr = pd.DataFrame(std_loss_arr)

    fig = go.Figure([go.Scatter(x=x, y=mean_loss_arr, mode="markers+lines", name="Mean Prediction",
                                line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=x, y=mean_loss_arr - 2 * std_loss_arr, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=x, y=mean_loss_arr + 2 * std_loss_arr, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)]
                    )

    fig.show()

    # fig = (go.Scatter(x=x, y=mean_loss_arr, mode="markers+lines", name="Mean Prediction",
    #                   line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
    #        go.Scatter(x=x, y=mean_loss_arr - 2 * std_loss_arr, fill=None, mode="lines",
    #                   line=dict(color="lightgrey"), showlegend=False),
    #        go.Scatter(x=x, y=mean_loss_arr + 2 * std_loss_arr, fill='tonexty', mode="lines",
    #                   line=dict(color="lightgrey"), showlegend=False))
