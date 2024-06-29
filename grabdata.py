import numpy as np
from PIL import Image
import random

inputs = np.load("inputs.npy", allow_pickle=True)
outputs = np.load("outputs.npy", allow_pickle=True)

def grabRandomData():
    index = random.randint(0, len(inputs))
    input = inputs[index][:, :, 1]
    im1 = Image.fromarray(input * 255)
    im1.show()
    outputs[index] += 10
    im2 = Image.fromarray(outputs[index] * 12)
    im2.show()