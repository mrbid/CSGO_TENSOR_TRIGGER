# James William Fletcher (james@voxdsp.com)
#   - CS:GO PewPew Trigger Bot v2 Prediction
#   https://github.com/tfcnn
#       July 2021
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from os.path import isfile
from os.path import getsize
from struct import pack
from time import sleep

# filter_resolution = 2
# model = Sequential([
#         keras.Input(shape=(28, 28, 3)),
#         layers.Conv2D(filter_resolution, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(filter_resolution*2, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(filter_resolution*4, kernel_size=(3, 3), activation="relu"),
#         layers.GlobalAveragePooling2D(),
#         layers.Flatten(),
#         layers.Dense(1, activation="sigmoid"),
# ])
# model.compile(optimizer='adam', loss='mean_squared_error')

# if isfile("weights.h5"):
#         model.load_weights("weights.h5")
#         print("weights loaded")

input_size = 2352

model = keras.models.load_model("keras_model")
input_size_floats = input_size*4
while True:
        sleep(0.001)
        if isfile("input.dat") and getsize("input.dat") == input_size_floats:
                with open("input.dat", 'rb') as f:
                        data = np.fromfile(f, dtype=np.float32)
                        if data.size == input_size:
                                input = np.reshape(data, [-1, 28,28,3])
                                r = model.predict(input)
                                with open("r.dat", "wb") as f2:
                                        f2.write(pack('f', r))

