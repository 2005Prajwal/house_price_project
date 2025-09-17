# task1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

CSV_PATH = "house_prices.csv"
MODEL_PATH = "best_model.pkl"

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation (only if enough training samples)
    cv_folds = min(5, len(X_train))
    if cv_folds > 1:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_mean = cv_scores.mean()
    else:
        cv_scores = []
        cv_mean = None

    print(f"\n📊 Results for {model_name}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    if cv_mean is not None:
        print(f"CV Mean R²: {cv_mean:.4f}")
    else:
        print("CV skipped (not enough data)")

    return model, y_pred, r2

def main():
    # 1) Load dataset
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print("✅ Loaded dataset. First 5 rows:")
    print(df.head())

    # 2) Features & target
    feature_cols = ["sqft", "bedrooms", "bathrooms"]
    X = df[feature_cols]
    y = df["price"]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_r2 = -np.inf
    best_pred = None

    # 5) Train & evaluate models
    for name, model in models.items():
        trained_model, y_pred, r2 = evaluate_model(model, X_train, X_test, y_train, y_test, model_name=name)
        if r2 > best_r2:
            best_r2 = r2
            best_model = trained_model
            best_pred = y_pred

    print(f"\n🏆 Best model: {type(best_model).__name__} (R² = {best_r2:.4f})")

    # 6) Visualization: Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, best_pred, color="blue", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color="red", linestyle="--", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)
    plt.show()

    # 7) Save the best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Best model saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
