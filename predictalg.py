from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

pdensity = np.array(Image.open("popdensity.png"))[:,:,0] / 255 * 2.4 # measured in hundreds of thousands per sq mile
vegetation = np.array(Image.open("vegetation.png"))[:,:,0] / 255 * 0.8
water = np.array(Image.open("water.png"))[:,:,0] / 255
applicable = np.array(Image.open("applicable_nojfk.png"))[:,:,0] / 255

def zerocenter(array):
    avg = np.sum(array) / array.size
    array -= avg
    return array

class block:
    def __init__(self):
        self.set = False
        self.values = np.zeros((24, 24))
    def getAvgLeft(self):
        return np.sum(self.values[:,0]) / 24
    def getAvgRight(self):
        return np.sum(self.values[:,23]) / 24
    def getAvgTop(self):
        return np.sum(self.values[0,:]) / 24
    def getAvgBottom(self):
        return np.sum(self.values[23,:]) / 24

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
model.load_weights("model.weights.h5")

def predict(pdensity, vegetation, water, applicable):
    height = pdensity.shape[0]
    width = pdensity.shape[1]
    hblocks = height // 23
    wblocks = width // 23
    blockset = [[block() for i in range(wblocks)] for j in range(hblocks)]
    for i in range(hblocks):
        for j in range(wblocks):
            if (i*23 -13 >= 0 and i*23 + 37 <= height and j*23 -13 >= 0 and j*23 + 37 <= width and np.sum(applicable[i*23:i*23+24, j*23:j*23+24]) > 288):
                blockset[i][j].set = True
                input = np.stack((np.copy(pdensity[i*23-13:i*23+37, j*23-13:j*23+37]),
                                  np.copy(vegetation[i*23-13:i*23+37, j*23-13:j*23+37]),
                                  np.copy(water[i*23-13:i*23+37, j*23-13:j*23+37]),
                                  np.copy(applicable[i*23-13:i*23+37, j*23-13:j*23+37])), axis=-1)
                blockset[i][j].values = np.multiply(model.predict(np.array([input])).reshape(24, 24), applicable[i*23:i*23+24, j*23:j*23+24])

    output = np.zeros((height, width))
    for i in range(hblocks):
        for j in range(wblocks):
            if blockset[i][j].set:
                output[i*23:i*23+24, j*23:j*23+24] = blockset[i][j].values

    im = Image.fromarray((output + 10) * 12)
    im.show()
