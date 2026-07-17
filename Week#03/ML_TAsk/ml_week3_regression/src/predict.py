
import os
import argparse
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def predict_sales(tv: float, radio: float, newspaper: float) -> float:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    new_data = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "radio", "newspaper"])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sales from ad budgets")
    parser.add_argument("--tv", type=float, required=True, help="TV advertising budget ($1000s)")
    parser.add_argument("--radio", type=float, required=True, help="Radio advertising budget ($1000s)")
    parser.add_argument("--newspaper", type=float, required=True, help="Newspaper advertising budget ($1000s)")
    args = parser.parse_args()

    result = predict_sales(args.tv, args.radio, args.newspaper)
    print(f"Predicted Sales for TV=${args.tv}k, Radio=${args.radio}k, Newspaper=${args.newspaper}k: {result:.2f} (in 1000 units)")
