#AUTOENCODER

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Define the Autoencoder architecture
input_dim = x_train.shape[1]  # 28 * 28 = 784
encoding_dim = 128  # Increased dimension of the encoded representation

# Input layer
input_img = Input(shape=(input_dim,))

# Encoder layers (add extra layers for a deeper network)
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)  # Another hidden layer
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(0.001))(encoded)

# Decoder layers (add layers to match encoder)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Use 'sigmoid' for normalized pixel values (0-1)

# Construct the Autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model using 'mean_squared_error' for continuous pixel values
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the Autoencoder
autoencoder.fit(x_train, x_train, epochs=7, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Evaluate the model (compute test loss)
test_loss = autoencoder.evaluate(x_test, x_test)
print(f'Test Loss: {test_loss:.4f}')

# Encode and decode the test data
decoded_imgs = autoencoder.predict(x_test)

# Visualize the original and reconstructed images
n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))

# Display original images
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()
