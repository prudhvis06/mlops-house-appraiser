import os
import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    data_path = os.path.join("data", "housing.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}. Run 'dvc pull' first!")

    data = pd.read_csv(data_path)
    num_data = data.select_dtypes(include=[np.number]).dropna()

    X = num_data.drop(["median_house_value"], axis=1)
    y = num_data["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    mlflow.set_experiment("House_Appraiser_Prices")

    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)

        print(f"📊 Training Finished [n_estimators={n_estimators}, max_depth={max_depth}] -> RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R2: {r2:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        model_path = os.path.join(models_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(rf, f)
        
        mlflow.sklearn.log_model(rf, "model")
        print(f"💾 Saved binary model artifact to {model_path}")

if __name__ == "__main__":
    main()
