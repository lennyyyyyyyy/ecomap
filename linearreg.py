import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
from PIL import Image

inputs = np.load("inputs.npy", allow_pickle=True)
outputs = np.load("outputs.npy", allow_pickle=True)
print(len(inputs))
inputs = np.array(inputs)
outputs = np.array(outputs)
inputs = inputs.reshape(inputs.shape[0], -1)
outputs = outputs.reshape(outputs.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(mean_squared_error(y_test, y_pred))

def grabRandomData():
    index = random.randint(0, len(inputs))
    outputs[index] += 10
    im2 = Image.fromarray(outputs[index].reshape(24, 24) * 12)
    im2.show()
    im3 = Image.fromarray(12*(regressor.predict(inputs[index].reshape(1, -1)).reshape(24, 24) + 10))
    im3.show()
