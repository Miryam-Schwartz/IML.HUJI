import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename).dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] > -20]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:\\Users\\Miryam\\Documents\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df['Year'] = df['Year'].astype(str)
    df_israel = df[df['Country'] == "Israel"]

    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color="Year",
                     title="Average Daily Temperature Change as a Function of the `DayOfYear`")
    fig.show()

    std_temp = df_israel.groupby('Month').agg({'Temp': 'std'})
    fig = px.bar(std_temp, x=list(range(1, 13)), y='Temp',
                 title=" Standard Deviation of Daily Temperatures as Function of Month",
                 labels={'x': 'Month', 'Temp': 'STD'})
    fig.show()

    # Question 3 - Exploring differences between countries
    df_county_month_mean = df.groupby(['Country', 'Month'])['Temp'].mean().reset_index()
    df_county_month_std = df.groupby(['Country', 'Month'])['Temp'].std().reset_index()
    fig = px.line(pd.DataFrame({'Mean_Temp': df_county_month_mean['Temp'],
                                'STD_Temp': df_county_month_std['Temp'],
                                'Country': df_county_month_mean['Country'],
                                'Month': df_county_month_mean['Month']}),
                  x='Month', y='Mean_Temp', color='Country', error_y='STD_Temp',
                  title="Average Monthly Temperature Colored by Country")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    x = df_israel['Month']
    y = df_israel['Temp']
    train_x, train_y, test_x, test_y = split_train_test(x, y)
    K = list(range(1, 11))
    loss = []
    for k in K:
        estimator = PolynomialFitting(k)
        estimator.fit(train_x.to_numpy(), train_y.to_numpy())
        loss.append(round(estimator.loss(test_x.to_numpy(), test_y.to_numpy()), 2))
    print("Test error recorded for each value of k:")
    print(loss)
    fig = px.bar(loss, x=list(range(1, 11)), y=loss,
                 title=" Test Error Recorded for Each Value of Polynom Degree",
                 labels={'x': 'k - Polynom Degree', 'y': 'Loss'})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    estimator = PolynomialFitting(5)
    estimator.fit(x, y)
    df = df[df['Country'] != 'Israel']
    countries = np.unique(df['Country'])
    loss = []
    for country in countries:
        temp_df = df[df['Country'] == country]
        loss.append(estimator.loss(temp_df['Month'], temp_df['Temp']))
    fig = px.bar(loss, x=countries, y=loss,
                 title="Israel's Model’s Error Over Each of the Other Countries",
                 labels={'x': 'Country', 'y': 'Loss'})
    fig.show()
