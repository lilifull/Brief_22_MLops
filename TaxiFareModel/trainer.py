from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from utils import compute_rmse
from encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from data import get_data, clean_data
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


class Trainer():

    def __init__(self, X,y):
       self.X = X
       self.y = y 
       self.experiment_name = 'test_experiment'
       self.mlflow_client = MlflowClient()

    def set_pipeline(self):
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder()),('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer(transformers=[
                                ('distance', dist_pipe,['pickup_latitude', 'pickup_longitude',
                                'dropoff_latitude', 'dropoff_longitude']),
                                ('time', time_pipe, ['pickup_datetime'])])
        pipeline = Pipeline([('preproc',preproc_pipe),
                ('linear_model', LinearRegression())])
        return pipeline
    
    def run(self):
        pipeline = self.set_pipeline()
        '''returns a trained pipelined model'''
        trained_pipeline = pipeline.fit(self.X, self.y)
        return trained_pipeline
    
    def evaluate(self,X_test, y_test, trained_pipeline):
        '''returns the value of the RMSE'''
        y_pred = trained_pipeline.predict(X_test)
        y_test = y_test.values
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":

    raw_data = get_data()
    df = clean_data(raw_data)

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15,random_state=42)

    trainer = Trainer(X_train, y_train)

    trained_pipeline = trainer.run()
    
    rmse = trainer.evaluate(X_val, y_val, trained_pipeline)

    model = trained_pipeline.steps[1][0]
    
    trainer.mlflow_log_param("model", model)
    trainer.mlflow_log_metric( "rmse",  rmse)

