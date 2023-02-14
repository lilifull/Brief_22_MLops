from sklearn.model_selection import train_test_split
from data import get_data, clean_data
from trainer import Trainer

raw_data = get_data()
df = clean_data(raw_data)

y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15,random_state=42)

trainer = Trainer()

pipeline = trainer.set_pipeline()
trainer.run(X_train, y_train,pipeline)
trainer.evaluate(X_val, y_val, pipeline)