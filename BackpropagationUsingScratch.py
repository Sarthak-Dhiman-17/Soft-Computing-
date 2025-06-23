#BACKPROPAGATION

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# One-hot encoding function for the target labels
def one_hot_encoding(y, num_classes):
    return np.eye(num_classes)[y]

# Neural Network class with manual backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def feedforward(self, X):
        # Forward pass (feedforward)
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_activation)

        return self.output

    def backpropagation(self, X, y, learning_rate):
        # Feedforward
        output = self.feedforward(X)

        # Compute output error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        # Compute hidden layer error
        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_layer_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagation(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.feedforward(X)))
                print(f'Epoch {epoch}, Loss: {loss:.6f}')

    def predict(self, X):
        output = self.feedforward(X)
        return np.argmax(output, axis=1)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert labels to one-hot encoding
y_one_hot = one_hot_encoding(y, 3)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Initialize the neural network
input_size = X_train.shape[1]  # Number of features (4)
hidden_size = 10               # Number of neurons in the hidden layer
output_size = y_one_hot.shape[1]  # Number of classes (3)

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
epochs = 10000
learning_rate = 0.01
nn.train(X_train, y_train, epochs, learning_rate)

# Predict and evaluate the model on validation data
y_val_pred = nn.predict(X_val)
y_val_actual = np.argmax(y_val, axis=1)

accuracy = accuracy_score(y_val_actual, y_val_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
