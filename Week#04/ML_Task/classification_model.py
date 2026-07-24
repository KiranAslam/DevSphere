import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Classification model trained successfully.")
    print("Accuracy score:", round(accuracy, 4))

    print("\nSample predictions:")
    for actual, predicted in zip(y_test[:10], predictions[:10]):
        print(f"Actual: {actual} | Predicted: {predicted}")

    print("\nWorkflow summary:")
    print("1. Load a classification dataset")
    print("2. Split data into train and test sets")
    print("3. Scale the features")
    print("4. Train a logistic regression classifier")
    print("5. Evaluate the model using accuracy")


if __name__ == "__main__":
    main()
