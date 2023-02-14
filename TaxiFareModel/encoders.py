from sklearn.base import BaseEstimator, TransformerMixin
from utils import haversine_vectorized, extract_time_features

# create distance column
class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['distance'] = haversine_vectorized(X)
    
        return X[['distance']]
    

# Create dow, hour, month, year, columns
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self,time_column='pickup_datetime'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = extract_time_features(X)
        return X[['dow','hour', 'month', 'year']]