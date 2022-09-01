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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def normaliseImage(arr):
    arr = arr.flatten()
    newarr = []
    for x in arr:
        if x != 0:
            newarr.append(x/255)
        else:
            newarr.append(0)
    return newarr

model = keras.models.load_model("keras_model")
while True:
        try:
                sleep(0.001)
                if isfile("/dev/shm/pred_input.dat") and getsize("/dev/shm/pred_input.dat") == 2365:
                        img = keras.preprocessing.image.load_img("/dev/shm/pred_input.dat")
                        arr = keras.preprocessing.image.img_to_array(img)
                        data = normaliseImage(np.array(arr))
                        input = np.reshape(data, [-1, 28,28,3])
                        r = model.predict(input, verbose=0)
                        with open("/dev/shm/pred_r.dat", "wb") as f2:
                                f2.write(pack('f', r))
        except Exception:
                pass
