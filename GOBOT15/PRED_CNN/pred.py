# James William Fletcher (github.com/mrbid)
#       C to ONNX Bridge for Predictor
#               DECEMBER 2022
import os
import numpy as np
import onnxruntime as rt
from os.path import isfile
from os.path import getsize
from os import remove
from struct import pack
from time import sleep

print("Don't close this window, it's running the C to ONNX `/dev/shm` bridge.")
print("\nYou may want to set this processes affinity or nice to prevent it going too buck wild on CPU resources.")

input_size = 784

sess = rt.InferenceSession("model.onnx") 
input_size_floats = input_size*4
while True:
        try:
                sleep(0.001)
                if isfile("/dev/shm/pred_input.dat") and getsize("/dev/shm/pred_input.dat") == input_size_floats:
                        with open("/dev/shm/pred_input.dat", 'rb') as f:
                                data = np.fromfile(f, dtype=np.float32)
                                f.close()
                                remove("/dev/shm/pred_input.dat")
                                if data.size == input_size:
                                        input = np.reshape(data, [-1, 28,28,1])
                                        input_name = sess.get_inputs()[0].name
                                        r = sess.run(None, {input_name: input})[0]
                                        with open("/dev/shm/pred_r.dat", "wb") as f2:
                                                f2.write(pack('f', r))
                                                f2.close()
        except Exception:
                pass
