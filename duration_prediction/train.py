#!/usr/bin/env python
# coding: utf-8

import pickle
import logging
from datetime import date

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


logger = logging.getLogger(__name__)


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read and preprocess a parquet file containing taxi trip data.

    This function reads a parquet file specified by the filename, computes the duration of each trip in minutes, 
    and filters the data to include trips with durations between 1 and 60 minutes. It also converts specific 
    categorical columns to string types.

    Parameters:
    filename (str): The path to the parquet file containing taxi trip data.

    Returns:
    pandas.DataFrame: A DataFrame with preprocessed taxi trip data.

    Note:
    The function assumes that the input file has columns 'lpep_dropoff_datetime' and 'lpep_pickup_datetime' 
    to calculate the 'duration' and 'PULocationID', 'DOLocationID' as categorical columns.
    """
    try:
        logger.info(f'reading data from {filename}...')
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise


def train(train_month: date, val_month: date, model_output_path: str) -> None:
    """
    Train a linear regression model to predict taxi trip duration.

    This function trains a linear regression model using taxi trip data from specified training and validation 
    months. The model is trained using a combination of categorical and numerical features and is evaluated 
    using the root mean square error (RMSE) metric. The trained model is then saved to the specified output path.

    Parameters:
    train_month (date): The month and year to use for training data.
    val_month (date): The month and year to use for validation data.
    model_output_path (str): Path to save the trained model.

    Returns:
    None: The function prints the RMSE of the model on the validation set and saves the trained model to disk.

    Note:
    The function assumes the availability of taxi trip data in parquet format hosted at a specified URL template.
    It requires 'PULocationID', 'DOLocationID', and 'trip_distance' as features, and 'duration' as the target variable.
    """
    url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = url_template.format(year=train_month.year, month=train_month.month)
    val_url = url_template.format(year=val_month.year, month=val_month.month)
    
    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    logger.debug(f'turning dataframes into dictionaries...')

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    logger.debug(f'number of records in train: {len(train_dicts)}')
    logger.debug(f'number of records in validation {len(val_dicts)}')

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    logger.debug(f'training the model...')
    pipeline.fit(train_dicts, y_train)

    y_pred = pipeline.predict(val_dicts)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    logger.info(f'rmse = {rmse}')

    logger.info(f'saving the model to {model_output_path}...')

    with open(model_output_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)