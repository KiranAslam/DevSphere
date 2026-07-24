import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros((1, output_dim))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _tanh(self, x):
        return np.tanh(x)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._softmax(z2)
        return z1, a1, z2, a2

    def _backward(self, X, y_one_hot, z1, a1, a2):
        m = X.shape[0]

        dz2 = a2 - y_one_hot
        dW2 = a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=3000):
        y_one_hot = np.eye(self.output_dim)[y]

        for epoch in range(epochs):
            z1, a1, z2, a2 = self._forward(X)
            dW1, db1, dW2, db2 = self._backward(X, y_one_hot, z1, a1, a2)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if epoch % 500 == 0:
                loss = -np.sum(np.log(a2[np.arange(len(y)), y])) / len(y)
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict(self, X):
        _, _, _, probabilities = self._forward(X)
        return np.argmax(probabilities, axis=1)


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SimpleNeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=8,
        output_dim=len(np.unique(y)),
        learning_rate=0.1,
    )

    model.train(X_train, y_train, epochs=4000)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining complete.")
    print("Accuracy on test set:", round(accuracy, 4))

    print("\nSample predictions:")
    for actual, predicted in zip(y_test[:10], predictions[:10]):
        print(f"Actual: {actual} | Predicted: {predicted}")

    print("\nWorkflow summary:")
    print("1. Load the Iris dataset")
    print("2. Normalize the input features")
    print("3. Build a simple neural network with one hidden layer")
    print("4. Train the model using backpropagation")
    print("5. Evaluate predictions on unseen test data")


if __name__ == "__main__":
    main()
