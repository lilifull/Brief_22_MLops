import pandas as pd


def get_data(nrows=10000):
    df = pd.read_csv('TaxiFareModel/data/train.csv', nrows=nrows)
    return df

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df[df.passenger_count.between(1, 10)]
    df = df[df.fare_amount.between(0, 60)]
    df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
    df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
    df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
    df = df[df["dropoff_longitude"].between(left = -74.3, right = -72.9 )]
    return df

