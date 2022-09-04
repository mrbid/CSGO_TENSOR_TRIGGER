# James William Fletcher (github.com/mrbid)
#       C to Keras Bridge for Predictor
#               SEPTEMBER 2022
import os
import numpy as np
from tensorflow import keras
from os.path import isfile
from os.path import getsize
from os import remove
from struct import pack
from time import sleep

print("Don't close this window, it's running the C to Keras `/dev/shm` bridge.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

input_size = 784

model = keras.models.load_model("keras_model")
input_size_floats = input_size*4
while True:
        try:
                sleep(0.001)
                if isfile("/dev/shm/pred_input.dat") and getsize("/dev/shm/pred_input.dat") == input_size_floats:
                        with open("/dev/shm/pred_input.dat", 'rb') as f:
                                data = np.fromfile(f, dtype=np.float32)
                                remove("/dev/shm/pred_input.dat")
                                if data.size == input_size:
                                        input = np.reshape(data, [-1, input_size])
                                        r = model.predict(input, verbose=0)
                                        with open("/dev/shm/pred_r.dat", "wb") as f2:
                                                f2.write(pack('f', r))
        except Exception:
                pass
