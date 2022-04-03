import sys
sys.path.append("G:\האחסון שלי\year2\semesterB\iml\IML.HUJI")
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).dropna(how='all').drop_duplicates()

    #convert date column to datetime
    full_data['booking_datetime'] = pd.to_datetime(
        full_data['booking_datetime'])
    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])


    # booking_datetime
    full_data['booking_datetime_weekday'] = full_data[
        'booking_datetime'].dt.day_of_week
    full_data['booking_datetime_month'] = pd.DatetimeIndex(
        full_data['booking_datetime']).month
    full_data['booking_datetime_hour'] = pd.DatetimeIndex(
        full_data['booking_datetime']).hour
    full_data['dist_booking_to_checkin'] = (full_data['checkin_date'] -
                                            full_data[
                                                'booking_datetime']).dt.days + 1

    # checkin_date, checkout_date
    full_data['checkin_date_weekday'] = full_data[
        'checkin_date'].dt.day_of_week
    full_data['checkin_date_month'] = pd.DatetimeIndex(
        full_data['checkin_date']).month
    full_data['length_of_vacation'] = (
                full_data['checkout_date'] - full_data['checkin_date']).dt.days

    # Categorical feature
    full_data = pd.get_dummies(full_data, columns=['hotel_country_code',
                                                   'accommadation_type_name',
                                                   'charge_option',
                                                   'guest_nationality_country_name',
                                                   'original_payment_type'])

    # h_customer_id
    full_data['num_of_booking_per_person'] = \
    full_data.groupby('h_customer_id')['h_customer_id'].transform('count')


    # True/False features
    full_data['is_user_logged_in'] = (full_data['is_user_logged_in']).astype(
        int)
    full_data['is_first_booking'] = (full_data['is_first_booking']).astype(int)
    full_data['request_latecheckin'] = (
                full_data['request_latecheckin'] == 1).astype(int)
    full_data['request_airport'] = (full_data['request_airport'] == 1).astype(
        int)
    full_data['request_earlycheckin'] = (
                full_data['request_earlycheckin'] == 1).astype(int)
    full_data['has_brand_code'] = (
        full_data['hotel_brand_code'].notnull()).astype(int)
    full_data['has_chain_code'] = (
        full_data['hotel_chain_code'].notnull()).astype(int)

    # labels
    if 'cancellation_datetime' in set(full_data.columns):
        labels = (full_data['cancellation_datetime'].notnull()).astype(int) # add between 2 dates
        full_data = full_data.drop(['cancellation_datetime'], axis=1)
    else:
        labels = 0


    #drop useless columns
    full_data = full_data.drop(
        ['h_booking_id', 'booking_datetime', 'checkin_date', 'checkout_date',
         'hotel_live_date', 'h_customer_id', 'customer_nationality',
         'origin_country_code', 'language', 'original_payment_method',
         'original_payment_currency', 'request_nonesmoke', 'request_highfloor',
         'request_largebed', 'request_twinbeds',
         'hotel_area_code', 'hotel_brand_code', 'hotel_chain_code', 'cancellation_policy_code'], axis=1)



    # columns that we ignored:
    # cancellation_policy_code
    # hotel_city_code
    # hotel_area_code

    return full_data, labels

def match_data_test(test: pd.DataFrame, train: pd.DataFrame):
    # Get missing columns in the training test
    missing_cols = set(train.columns) - set(test.columns)
    # Add a missing column in test set with default value equal to 0

    # zero_cols = np.zeros(shape=(test.shape[0], len(missing_cols)))
    # d = pd.DataFrame(zero_cols, columns=missing_cols)
    # test = pd.concat([test, d], axis=1, ignore_index=True)

    for c in missing_cols:
        test[c] = 0

    #remove spare cols from the test
    cols_in_test = set(test.columns) - set(train.columns)
    test = test.drop(cols_in_test, axis=1)
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]
    return test

def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    train, train_cancellation_labels =\
        load_data("C:/Users/Miryam/Documents/IML.HUJI/datasets/agoda_cancellation_train.csv")
    print(train.shape)

    #train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    #print(train_X)

    test, temp = load_data("C:/Users/Miryam/Documents/IML.HUJI/challenge/test_data/test_set_week_1.csv")

    # Get missing columns in the training test
    missing_cols = set(train.columns) - set(test.columns)
    # Add a missing column in test set with default value equal to 0

    # zero_cols = np.zeros(shape=(test.shape[0], len(missing_cols)))
    # d = pd.DataFrame(zero_cols, columns=missing_cols)
    # test = pd.concat([test, d], axis=1, ignore_index=True)

    for c in missing_cols:
        test[c] = 0

    #remove spare cols from the test
    cols_in_test = set(test.columns) - set(train.columns)
    test = test.drop(cols_in_test, axis=1)
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]

    # test = match_data_test(test, train)

    print(test.shape)


    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train, train_cancellation_labels)

    # Store model predictions over test set
    evaluate_and_export(estimator, test, "208671180_318508439_314809682.csv")