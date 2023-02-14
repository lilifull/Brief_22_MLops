from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from utils import compute_rmse
from encoders import DistanceTransformer, TimeFeaturesEncoder

class Trainer():
         
    def set_pipeline(self):
        pipe = Pipeline(steps=[('preproc',
                 ColumnTransformer(transformers=[('distance',
                                                  Pipeline(steps=[('dist_trans',
                                                                   DistanceTransformer()),
                                                                  ('stdscaler',
                                                                   StandardScaler())]),
                                                  ['pickup_latitude',
                                                   'pickup_longitude',
                                                   'dropoff_latitude',
                                                   'dropoff_longitude']),
                                                 ('time',
                                                  Pipeline(steps=[('time_enc',
                                                                   TimeFeaturesEncoder(time_column='pickup_datetime')),
                                                                  ('ohe',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                  ['pickup_datetime'])])),
                ('linear_model', LinearRegression())])
        return pipe
    
    def run(self,X_train, y_train, pipeline):
        '''returns a trained pipelined model'''
        pipeline = pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self,X_test, y_test, pipeline):
        '''returns the value of the RMSE'''
        y_pred = pipeline.predict(X_test)
        y_test = y_test.values
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


