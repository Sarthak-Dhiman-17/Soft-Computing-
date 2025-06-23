#Experiment 7
#BACKPROPAGATION

import numpy as np
import matplotlib.pyplot as plt
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

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        # Lists to store loss and accuracy for each epoch
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            # Backpropagation and feedforward for training data
            self.backpropagation(X_train, y_train, learning_rate)

            # Calculate training loss and accuracy
            train_output = self.feedforward(X_train)
            train_loss = np.mean(np.square(y_train - train_output))
            train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_output, axis=1))

            # Calculate validation loss and accuracy
            val_output = self.feedforward(X_val)
            val_loss = np.mean(np.square(y_val - val_output))
            val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_output, axis=1))

            # Store the results
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        return train_losses, val_losses, train_accuracies, val_accuracies

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
epochs = 1000
learning_rate = 0.01
train_losses, val_losses, train_accuracies, val_accuracies = nn.train(X_train, y_train, X_val, y_val, epochs, learning_rate)

# Predict and evaluate the model on validation data
y_val_pred = nn.predict(X_val)
y_val_actual = np.argmax(y_val, axis=1)
accuracy = accuracy_score(y_val_actual, y_val_pred)
print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

# Plot the losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label="Training Loss")
plt.plot(range(epochs), val_losses, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot the accuracies
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label="Training Accuracy")
plt.plot(range(epochs), val_accuracies, label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
