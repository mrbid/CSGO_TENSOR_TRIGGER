# James William Fletcher (github.com/mrbid)
#   August 2022
import os
import sys
import glob
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from time import time_ns
from sys import exit
from os.path import isdir
from os.path import isfile
from os import mkdir

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
project = "aim_model"
training_iterations = 128
filter_resolution = 16
batches = 24

tc = len(glob.glob('target/*'))     # target sample count/length
ntc = len(glob.glob('nontarget/*')) # non-target sample count/length

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

# def featurewisenorm(a):
#     layer = keras.layers.experimental.preprocessing.Normalization()
#     layer.adapt(a)
#     return layer(a).numpy()

# def meanNormaliseImage(arr):
#     arr = arr.flatten()
#     newarr = []
#     len, ofs, rm, gm, bm, rh, gh, bh = 0,0,0,0,0,0,0,0
#     rl, gl, bl = 99999999999999,99999999999999,99999999999999
#     for x in arr:
#         if ofs == 0:
#             if x > rh: rh = x
#             if x < rl: rl = x
#             rm += x
#         elif ofs == 1:
#             if x > gh: gh = x
#             if x < gl: gl = x
#             gm += x
#         elif ofs == 2:
#             if x > bh: bh = x
#             if x < bl: bl = x
#             bm += x
#         len += 1
#         ofs += 1
#         if ofs == 3: ofs = 0
#     clen = len/3
#     rm /= clen
#     gm /= clen
#     bm /= clen
#     rmd = rh-rl
#     gmd = gh-gl
#     bmd = bh-bl
#     ofs, len = 0,0
#     for x in arr:
#         if ofs == 0:
#             newarr.append( ((x-rm)+1e-7) / (rmd+1e-7) )
#         elif ofs == 1:
#             newarr.append( ((x-gm)+1e-7) / (gmd+1e-7) )
#         elif ofs == 2:
#             newarr.append( ((x-bm)+1e-7) / (bmd+1e-7) )
#         len += 1
#         ofs += 1
#         if ofs == 3: ofs = 0
#     return newarr

# load training data
if isdir(project):
    nontargets_x = []
    nontargets_y = []
    if isfile(project + "/nontargets_x.npy"):
        print("Loading nontargets_x dataset..")
        st = time_ns()
        nontargets_x = np.load(project + "/nontargets_x.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    else:
        print("Creating nontargets_x dataset..")
        st = time_ns()
        files = glob.glob("nontarget/*")
        for f in files:
            img = keras.preprocessing.image.load_img(f)
            arr = keras.preprocessing.image.img_to_array(img)
            arr = np.array(arr)
            #print("before:", arr)
            nontargets_x.append(normaliseImage(arr))
            #print("after:", nontargets_x)
            #exit()
        nontargets_x = np.reshape(nontargets_x, [ntc, 28,28,3])
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
        print("Loading nontargets_x dataset..")
        st = time_ns()
        targets_x = np.load(project + "/targets_x.npy")
        print("Done in {:.2f}".format((time_ns()-st)/1e+9) + " seconds.")
    else:
        print("Creating targets_x dataset..")
        st = time_ns()
        files = glob.glob("target/*")
        for f in files:
            img = keras.preprocessing.image.load_img(f)
            arr = keras.preprocessing.image.img_to_array(img)
            arr = np.array(arr)
            targets_x.append(normaliseImage(arr))
        targets_x = np.reshape(targets_x, [tc, 28,28,3])
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
model = Sequential([
        keras.Input(shape=(28, 28, 3)),
        layers.Conv2D(filter_resolution, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filter_resolution*2, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filter_resolution*4, kernel_size=(3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
])

# output summary
model.summary()

# optim = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam', loss='mean_squared_error')


# train network
st = time_ns()
model.fit(train_x, train_y, epochs=training_iterations, batch_size=batches)
# model.fit(x_train, y_train, epochs=training_iterations, batch_size=batches, validation_data=(x_val, y_val))
timetaken = (time_ns()-st)/1e+9
print("")
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")


# save model
if not isdir(project):
    mkdir(project)
if isdir(project):
    # save model
    model.save("../PredictBot/keras_model")
    f = open(project + "/model.txt", "w")
    if f:
        f.write(model.to_json())
    f.close()

    # save HDF5 weights
    model.save_weights(project + "/weights.h5")

    # save flat weights
    for layer in model.layers:
        if layer.get_weights() != []:
            f = open(project + "/" + layer.name + "_full.txt", "w")
            if f:
                f.write(str(layer.get_weights()))
            f.close()
            np.savetxt(project + "/" + layer.name + ".csv", layer.get_weights()[0].flatten(), delimiter=",") # weights
            np.savetxt(project + "/" + layer.name + "_bias.csv", layer.get_weights()[1].flatten(), delimiter=",") # biases

    # save CNN weights as C header
    f = open(project + "/" + project + "_layers.h", "w")
    if f:
        f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n")
        for layer in model.layers:
            isfirst = 0
            weights = layer.get_weights()
            if weights != []:
                f.write("const float " + layer.name + "[] = {")
                for w in weights[0].flatten():
                    if isfirst == 0:
                        f.write(str(w))
                        isfirst = 1
                    else:
                        f.write("," + str(w))
                f.write("};\n\n")
                isfirst = 0
                f.write("const float " + layer.name + "_bias[] = {")
                for w in weights[1].flatten():
                    if isfirst == 0:
                        f.write(str(w))
                        isfirst = 1
                    else:
                        f.write("," + str(w))
                f.write("};\n\n")
        f.write("#endif\n")
    f.close()


# show results
print("")
pt = model.predict(targets_x)
ptavg = np.average(pt)

pnt = model.predict(nontargets_x)
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