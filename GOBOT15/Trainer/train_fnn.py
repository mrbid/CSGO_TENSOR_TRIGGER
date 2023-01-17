# James William Fletcher (github.com/mrbid)
#   September 2022
import os
import sys
import glob
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from time import time_ns
from sys import exit
from os.path import isdir
from os.path import isfile
from os import mkdir

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# train only on CPU?
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
inputsize = 784
project = "aim_model_fnn"
training_iterations = 333
activator = 'tanh'
hidden_layers = 0
layer_units = 1024
batches = 24
use_bias = False
beef_nontargets = 333

tc =  len(glob.glob('target/*'))    # target sample count/length
ntc = len(glob.glob('nontarget/*')) # non-target sample count/length

# if non-targets is less than targets then padd/"beef" it out
if tc-ntc > 0:
    beef_nontargets = tc-ntc
    if beef_nontargets > 0:
        print("\n\n!!! Padding with " + str(beef_nontargets) + " random non-targets.\n")
        tc = tc
        ntc = tc

# make project directory
if not isdir(project):
    mkdir(project)

# helpers (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def normaliseImage(arr):
    arr = arr.flatten()
    newarr = []
    for x in arr:
        if x != 0:
            newarr.append(x/255)
        else:
            newarr.append(0)
    return newarr

# load training data
if isdir(project):
    nontargets_x = []
    nontargets_y = []
    if isfile(project + "/nontargets_x.npy"):
        print("Loading nontargets_x dataset.. (" + str(ntc) + ")")
        st = time_ns()
        nontargets_x = np.load(project + "/nontargets_x.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds. (" + "{:.0f}".format(nontargets_x.size/inputsize) + ")")
    else:
        print("Creating nontargets_x dataset.. (" + str(ntc) + ")")
        st = time_ns()
        files = glob.glob("nontarget/*")
        for f in files:
            img = keras.preprocessing.image.load_img(f, color_mode="grayscale")
            arr = keras.preprocessing.image.img_to_array(img)
            arr = np.array(arr)
            #print("before:", arr)
            nontargets_x.append(normaliseImage(arr))
            #print("after:", nontargets_x)
            #exit()
        nontargets_x = np.reshape(nontargets_x, [ntc, inputsize])
        np.save(project + "/nontargets_x.npy", nontargets_x)
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    if isfile(project + "/nontargets_y.npy"):
        print("Loading nontargets_y dataset..")
        st = time_ns()
        nontargets_y = np.load(project + "/nontargets_y.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    else:
        print("Creating nontargets_y dataset..")
        st = time_ns()
        nontargets_y = np.zeros([ntc, 1], dtype=np.float32)
        np.save(project + "/nontargets_y.npy", nontargets_y)
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")

    targets_x = []
    targets_y = []
    if isfile(project + "/targets_x.npy"):
        print("Loading targets_x dataset.. (" + str(tc) + ")")
        st = time_ns()
        targets_x = np.load(project + "/targets_x.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds. (" + "{:.0f}".format(targets_x.size/inputsize) + ")")
    else:
        print("Creating targets_x dataset.. (" + str(tc) + ")")
        st = time_ns()
        files = glob.glob("target/*")
        for f in files:
            img = keras.preprocessing.image.load_img(f, color_mode="grayscale")
            arr = keras.preprocessing.image.img_to_array(img)
            arr = np.array(arr)
            targets_x.append(normaliseImage(arr))
        targets_x = np.reshape(targets_x, [tc, inputsize])
        np.save(project + "/targets_x.npy", targets_x)
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    if isfile(project + "/targets_y.npy"):
        print("Loading targets_y dataset..")
        st = time_ns()
        targets_y = np.load(project + "/targets_y.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    else:
        print("Creating targets_y dataset..")
        st = time_ns()
        targets_y = np.ones([tc, 1], dtype=np.float32)
        np.save(project + "/targets_y.npy", targets_y)
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")

# print(targets_x.shape)
# print(nontargets_x.shape)
# exit()

train_x = np.concatenate((nontargets_x, targets_x), axis=0)
train_y = np.concatenate((nontargets_y, targets_y), axis=0)

# beef up with random non-target samples
if beef_nontargets > 0:
    add_x = np.random.uniform(low=0, high=1, size=(beef_nontargets, inputsize))
    add_y = np.zeros([beef_nontargets, 1], dtype=np.float32)
    train_x = np.concatenate((train_x, add_x), axis=0)
    train_y = np.concatenate((train_y, add_y), axis=0)

shuffle_in_unison(train_x, train_y)

# x_val = train_x[-230:]
# y_val = train_y[-230:]
# x_train = train_x[:-230]
# y_train = train_y[:-230]

# print(x_val.shape)
# print(y_val.shape)
# print(x_train.shape)
# print(y_train.shape)
# exit()

# print(y_train)
# exit()


# construct neural network
model = Sequential()
model.add(Dense(layer_units, activation=activator, use_bias=use_bias, input_dim=inputsize))

if hidden_layers > 0:
    for x in range(hidden_layers):
        model.add(Dense(layer_units, use_bias=use_bias, activation=activator))

model.add(Dense(1, use_bias=use_bias, activation='sigmoid'))

# output summary
model.summary()

# optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam', loss='mean_squared_error')


# train network
st = time_ns()
history = model.fit(train_x, train_y, epochs=training_iterations, batch_size=batches)
timetaken = (time_ns()-st)/1e+9
print("")
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")


# save info
if not isdir(project):
    mkdir(project)
if isdir(project):
    # save keras model
    model.save("../PRED_FNN/keras_model")

    # save json model
    f = open(project + "/model.txt", "w")
    if f:
        f.write(model.to_json())
    f.close()

    # save HDF5 weights
    model.save_weights(project + "/weights.h5")

    # save flat weights
    for layer in model.layers:
        if layer.get_weights() != []:
            np.savetxt(project + "/" + layer.name + ".csv", layer.get_weights()[0].flatten(), delimiter=",") # weights
            if use_bias == True:
                np.savetxt(project + "/" + layer.name + "_bias.csv", layer.get_weights()[1].flatten(), delimiter=",") # bias

    # save weights for C array
    print("")
    print("Exporting weights...")
    li = 0
    f = open(project + "/" + project + "_layers.h", "w")
    f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n// loss: " + "a{:E}".format(history.history['loss'][-1]) + "\n\n")
    if f:
        for layer in model.layers:
            total_layer_weights = layer.get_weights()[0].transpose().flatten().shape[0]
            total_layer_units = layer.units
            layer_weights_per_unit = total_layer_weights / total_layer_units
            #print(layer.get_weights()[0].flatten().shape)
            #print(layer.units)
            print("+ Layer:", li)
            print("Total layer weights:", total_layer_weights)
            print("Total layer units:", total_layer_units)
            print("Weights per unit:", int(layer_weights_per_unit))

            f.write("const float " + project + "_layer" + str(li) + "[] = {")
            isfirst = 0
            wc = 0
            bc = 0
            if layer.get_weights() != []:
                for weight in layer.get_weights()[0].transpose().flatten():
                    wc += 1
                    if isfirst == 0:
                        f.write(str(weight))
                        isfirst = 1
                    else:
                        f.write("," + str(weight))
                    if wc == layer_weights_per_unit:
                        if use_bias == True:
                            f.write(", /* bias */ " + str(layer.get_weights()[1].transpose().flatten()[bc]))
                            #print("bias", str(layer.get_weights()[1].flatten()[bc]))
                        wc = 0
                        bc += 1
            f.write("};\n\n")
            li += 1
    f.write("#endif\n")
    f.close()


# show results
print("")
pt = model.predict(targets_x, verbose=0)
ptavg = np.average(pt)

pnt = model.predict(nontargets_x, verbose=0)
pntavg = np.average(pnt)

cnzpt =  np.count_nonzero(pt <= pntavg)
cnzpts = np.count_nonzero(pt >= ptavg)
avgsuccesspt = (100/tc)*cnzpts
avgfailpt = (100/tc)*cnzpt
outlierspt = tc - int(cnzpt + cnzpts)

cnzpnt =  np.count_nonzero(pnt >= ptavg)
cnzpnts = np.count_nonzero(pnt <= pntavg)
avgsuccesspnt = (100/ntc)*cnzpnts
avgfailpnt = (100/ntc)*cnzpnt
outlierspnt = ntc - int(cnzpnts + cnzpnt)

print("training_iterations:", training_iterations)
print("activator:", activator)
if hidden_layers > 0:
    print("layers:", hidden_layers)
print("layer_units:", layer_units)
print("batches:", batches)
print("")
print("target:", "{:.0f}".format(np.sum(pt)) + "/" + str(tc))
print("target-max:", "{:.3f}".format(np.amax(pt)))
print("target-avg:", "{:.3f}".format(ptavg))
print("target-min:", "{:.3f}".format(np.amin(pt)))
print("target-avg-success:", str(cnzpts) + "/" + str(tc), "(" + "{:.2f}".format(avgsuccesspt) + "%)")
print("target-avg-fail:", str(cnzpt) + "/" + str(tc), "(" + "{:.2f}".format(avgfailpt) + "%)")
print("target-avg-outliers:", str(outlierspt) + "/" + str(tc), "(" + "{:.2f}".format((100/tc)*outlierspt) + "%)")
print("")
print("nontarget:", "{:.0f}".format(np.sum(pnt)) + "/" + str(ntc))
print("nontarget-max:", "{:.3f}".format(np.amax(pnt)))
print("nontarget-avg:", "{:.3f}".format(pntavg))
print("nontarget-min:", "{:.3f}".format(np.amin(pnt)))
print("nontarget-avg-success:", str(cnzpnts) + "/" + str(ntc), "(" + "{:.2f}".format(avgsuccesspnt) + "%)")
print("nontarget-avg-fail:", str(cnzpnt) + "/" + str(ntc), "(" + "{:.2f}".format(avgfailpnt) + "%)")
print("nontarget-avg-outliers:", str(outlierspnt) + "/" + str(ntc), "(" + "{:.2f}".format((100/ntc)*outlierspnt) + "%)")
