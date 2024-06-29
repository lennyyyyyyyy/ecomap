from PIL import Image
import numpy as np
import random
import json
import math

pdensity = np.array(Image.open("popdensity.png"))[:,:,0] / 255 * 2.4 # measured in hundreds of thousands per sq mile
vegetation = np.array(Image.open("vegetation.png"))[:,:,0] / 255 * 0.8
water = np.array(Image.open("water.png"))[:,:,0] / 255
applicable = np.array(Image.open("applicable_nojfk.png"))[:,:,0] / 255
temps = np.load("temps.npy", allow_pickle=True) # variation in F, nonapplicable is marked as 0 by default


inputs = []
outputs = []

def zerocenter(array):
    avg = np.sum(array) / array.size
    array -= avg
    return array

for i in np.arange(0, 2345-50, 30):
    for j in np.arange(0, 2334-50, 30):
        applicable_square = np.copy(applicable[i:i+50, j:j+50])
        applicable_squares = np.sum(applicable_square)
        if applicable_squares > 1250: # 50% of the square is applicable
            pdensity_square = zerocenter(np.copy(pdensity[i:i+50, j:j+50]))
            vegetation_square = zerocenter(np.copy(vegetation[i:i+50, j:j+50]))
            water_square = np.copy(water[i:i+50, j:j+50])
            temps_square = zerocenter(np.copy(temps[i+13:i+37, j+13:j+37]))
            lastcombinedsquare = None
            for k in range(4):
                combined_square = np.stack((pdensity_square, vegetation_square, water_square, applicable_square), axis=-1)
                combined_squareT = np.stack((pdensity_square.T, vegetation_square.T, water_square.T, applicable_square.T), axis=-1)
                inputs.append(combined_square)
                outputs.append(temps_square)
                inputs.append(combined_squareT)
                outputs.append(temps_square.T)
                pdensity_square = np.rot90(pdensity_square)
                vegetation_square = np.rot90(vegetation_square)
                water_square = np.rot90(water_square)
                temps_square = np.rot90(temps_square)
                applicable_square = np.rot90(applicable_square)


# random.shuffle(inputs)
# random.shuffle(outputs)
np.save("inputs.npy", inputs)
np.save("outputs.npy", outputs)