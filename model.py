# 1st version of CNN
import sys
import numpy as np
import cv2
import csv

# we expect at least a training-set name, optionally the number of epochs
if len(sys.argv) < 2:
    print('usage:')
    print('   python {} <training-set-name> [epochs]'.format(sys.argv[0]))
    quit()

tsname = str(sys.argv[1])

epochs = 5 # small default for testing
if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

# read both the files that were meant for training and validation
# => unlike initially planned, we do the split later
lines = []
fn = '{}_train.csv'.format(tsname)
with open(fn) as csvfile:
    print('[info] reading {}'.format(fn))
    reader = csv.reader(csvfile)
    for l in reader:
        lines.append(l)

fn = '{}_valid.csv'.format(tsname)
with open(fn) as csvfile:
    print('[info] reading {}'.format(fn))
    reader = csv.reader(csvfile)
    for l in reader:
        lines.append(l)

print('[info] reading images/ building X/Y'.format(fn))
limg = []
langle = []
for l in lines:
    cfn = str(l[0])
    imgbgr = cv2.imread(cfn)
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB) # use RGB like drive.py
    limg.append(img)
    angle = float(l[3])
    langle.append(angle)

    # flip the image to double the data
    fimg = cv2.flip(img,flipCode=1)
    limg.append(fimg)
    fangle = -1.0 * angle
    langle.append(fangle)

    #cv2.imshow('img',img)
    #cv2.waitKey(10000)
    #cv2.imshow('img',fimg)
    #cv2.waitKey(10000)
    #quit()


X = np.array(limg)
del limg
Y = np.array(langle)
del langle

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D

# the actual model- building is here
model = Sequential()
model.add(Cropping2D(cropping=((30, 10), (0, 0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))          # -> 120x320x3
model.add(Conv2D(5,5,strides=3,activation='relu'))      # -> 39x106x5
model.add(MaxPooling2D(pool_size=(2,2)))                # -> 19x53x5
model.add(Conv2D(12,3,activation='relu'))               # -> 17x51x12
model.add(MaxPooling2D(pool_size=(2,2)))                # -> 8x25x12
model.add(Flatten(input_shape=(12,25,12)))              # -> 2400
model.add(Dense(711,activation='relu'))                 # -> 711
model.add(Dropout(0.5))                                 # -> 711
model.add(Dense(111,activation='relu'))                 # -> 111
model.add(Dropout(0.5))                                 # -> 111
model.add(Dense(1))                                     # -> 1      Tada!!!
print(model.summary())
#quit();

print('[info] creating and compiling model')
model.compile(loss='mse',optimizer='adam')

print('[info] !! fitting')
#model.fit(X,Y,validation_split=0.2,shuffle=False,epochs=epochs)
model.fit(X,Y,validation_split=0.2,shuffle=True,batch_size=67,epochs=epochs)

fn = '{}.cnn.model'.format(tsname)
print('[info] saving model to "{}"'.format(fn))
model.save(fn)
