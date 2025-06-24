
#RNN using SIMPLERNN

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense ,LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train,maxlen=max_length)
X_test = pad_sequences(X_test,maxlen=max_length)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
model.add(SimpleRNN(units=64, activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test,y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss : ",loss)
print("Test accuracy : ",accuracy)

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure(figsize=(12,4))
  plt.subplot(1,2,1)
  plt.plot(history.history['loss'], label="Training Loss")
  plt.plot(history.history['val_loss'], label="Validation Loss")
  plt.title("Model Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(['Train','Validation'],loc='upper right')

  plt.subplot(1,2,2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title("Model Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(['Train','Validaton'], loc='upper right')
  plt.show()


plot_history(history)

# Select a subset of the test data
subset_size = 100  # Choose the size of the subset
X_test_subset = X_test[:subset_size]
y_test_subset = y_test[:subset_size]

# Evaluate the model on the subset
loss, accuracy = model.evaluate(X_test_subset, y_test_subset)
print(f"Subset Test Accuracy: {accuracy*100:.2f}%")

# Make predictions on the subset
predictions = model.predict(X_test_subset)
# Convert predictions to binary outcomes
binary_predictions = (predictions > 0.5).astype("int32")

# Display first 10 predictions and actual labels for analysis
for i in range(10):
    print(f"Review {i+1}: Predicted: {binary_predictions[i][0]}, Actual: {y_test_subset[i]}")