# James William Fletcher (james@voxdsp.com)
#       C to Keras Bridge for Predictor
#               JULY 2021
import numpy as np
from tensorflow import keras
from os.path import isfile
from os.path import getsize
from os import remove
from struct import pack
from time import sleep

input_size = 2352

model = keras.models.load_model("keras_model")
input_size_floats = input_size*4
while True:
        sleep(0.001)
        if isfile("input.dat") and getsize("input.dat") == input_size_floats:
                with open("input.dat", 'rb') as f:
                        data = np.fromfile(f, dtype=np.float32)
                        remove("input.dat")
                        if data.size == input_size:
                                input = np.reshape(data, [-1, input_size])
                                r = model.predict(input)
                                with open("r.dat", "wb") as f2:
                                        f2.write(pack('f', r))

