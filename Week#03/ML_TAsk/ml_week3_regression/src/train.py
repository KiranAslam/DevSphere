
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_data, check_data_quality, split_features_target, get_train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "advertising.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
def evaluate(y_true, y_pred, label):
   
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2}
def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    check_data_quality(df)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    print(f"\nSaved correlation heatmap -> outputs/correlation_heatmap.png")
    X_simple = df[["TV"]]
    y = df["sales"]
    X_train_s, X_test_s, y_train_s, y_test_s = get_train_test_split(X_simple, y)
    simple_model = LinearRegression()
    simple_model.fit(X_train_s, y_train_s)
    y_pred_simple = simple_model.predict(X_test_s)
    simple_results = evaluate(y_test_s, y_pred_simple, "Simple Linear Regression (TV only)")
    X, y = split_features_target(df, target_col="sales")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    multi_model = LinearRegression()
    multi_model.fit(X_train_scaled, y_train)
    y_pred_multi = multi_model.predict(X_test_scaled)
    multi_results = evaluate(y_test, y_pred_multi, "Multiple Linear Regression (TV+radio+newspaper)")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": multi_model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    print("\nFeature importance (standardized coefficients):")
    print(coef_df)

    comparison = pd.DataFrame([simple_results, multi_results])
    comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison saved -> outputs/model_comparison.csv")
    print(comparison)
    predictions_df = pd.DataFrame({
        "Actual_Sales": y_test.values,
        "Predicted_Sales": np.round(y_pred_multi, 2),
        "Error": np.round(y_test.values - y_pred_multi, 2)
    })
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("\nPredictions saved -> outputs/predictions.csv")
    print(predictions_df.head(10))
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_multi, alpha=0.7, color="teal")
    lims = [min(y_test.min(), y_pred_multi.min()), max(y_test.max(), y_pred_multi.max())]
    plt.plot(lims, lims, color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales (Multiple Linear Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
    plt.close()
    print("Saved plot -> outputs/actual_vs_predicted.png")

    joblib.dump(multi_model, os.path.join(MODEL_DIR, "regression_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nModel saved -> models/regression_model.pkl")
    print(f"Scaler saved -> models/scaler.pkl")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
