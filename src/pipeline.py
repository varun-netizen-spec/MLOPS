import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump
import os

def generate_data():
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + np.random.randn(100) * 2
    return X, y

def train_model():
    X, y = generate_data()
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print("Model MSE:", mse)

    os.makedirs("model", exist_ok=True)
    dump(model, "model/model.pkl")

if __name__ == "__main__":
    train_model()

