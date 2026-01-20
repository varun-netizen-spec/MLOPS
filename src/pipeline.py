import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump
import os
import mlflow
import mlflow.sklearn

def generate_data():
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + np.random.randn(100) * 2
    return X, y

def train_model():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Mini-MLOps-Experiment")

    with mlflow.start_run():
        X, y = generate_data()

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        # 🔹 Log params & metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("samples", len(X))
        mlflow.log_metric("mse", mse)

        # 🔹 Save model
        os.makedirs("model", exist_ok=True)
        dump(model, "model/model.pkl")

        # 🔹 Log model as artifact
        mlflow.sklearn.log_model(model, "model")

        print("Model trained | MSE:", mse)

if __name__ == "__main__":
    train_model()
