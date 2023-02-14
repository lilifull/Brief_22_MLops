import numpy as np
import math
import pandas as pd


# Calculate the distance
def haversine_vectorized(df, 
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """ 
        Calculates the great circle distance between two points 
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df.
        Computes the distance in kms.
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# Extract hour, dow, month, year from pickup_datetime columns
def extract_time_features(df):
    """Extract hour, dow, month, year from pickup_datetime columns"""
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dow'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    return df


# Calculate RMSE 
def compute_rmse(y_pred, y_true):
    diff = []
    for i in range(len(y_pred)) : 
        a = (y_pred[i]-y_true[i])**2
        diff.append(a)
        
    RMSE = sum(diff)
    RMSE = RMSE/len(y_pred)
    RMSE =  math.sqrt(RMSE)
    return RMSE