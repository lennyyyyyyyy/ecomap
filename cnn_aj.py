import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

inputs = np.load("inputs.npy", allow_pickle=True)
outputs = np.load("outputs.npy", allow_pickle=True)

inputs = np.array(inputs)
outputs = np.array(outputs)

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)


# cnn

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 4)),
    MaxPooling2D((2, 2)), 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(576, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=20, batch_size=32, validation_split=0.2)

test_loss = model.evaluate(X_test, y_test.reshape(y_test.shape[0], -1))
print(f'Test loss: {test_loss}')