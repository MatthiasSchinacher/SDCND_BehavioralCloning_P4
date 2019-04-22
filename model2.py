# 2nd version of CNN
import sys
import numpy as np
import cv2
import csv

if len(sys.argv) < 2:
    print('usage:')
    print('   python {} <training-set-name> [epochs]'.format(sys.argv[0]))
    quit()

tsname = str(sys.argv[1])

epochs = 5
if len(sys.argv) > 2:
    epochs = int(sys.argv[2])

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

#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model = Sequential()
model.add(Cropping2D(cropping=((40, 20), (0, 0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))          # -> 100x320x3
model.add(Conv2D(7,3,strides=2,activation='relu'))      # -> 49x159x7
model.add(MaxPooling2D(pool_size=(5,5),strides=2))      # -> 23x78x7
model.add(Conv2D(12,(4,5),activation='relu'))           # -> 20x74x12
model.add(MaxPooling2D(pool_size=(2,3),strides=2))      # -> 10x36x12
model.add(Conv2D(24,(4,4),activation='relu'))           # -> 7x33x24
model.add(Flatten(input_shape=(7,33,24)))               # -> 5544
model.add(Dropout(0.5))                                 # -> 5544
model.add(Dense(911,activation='relu'))                 # -> 911
model.add(Dense(151,activation='relu'))                 # -> 151
model.add(Dense(31,activation='relu'))                  # -> 31
model.add(Dense(1))
print(model.summary())
#quit();

print('[info] creating and compiling model')
model.compile(loss='mse',optimizer='adam')

print('[info] !! fitting')
#model.fit(X,Y,validation_split=0.2,shuffle=False,epochs=epochs)
model.fit(X,Y,validation_split=0.2,shuffle=True,batch_size=67,epochs=epochs)

fn = '{}.cnn.model2'.format(tsname)
print('[info] saving model to "{}"'.format(fn))
model.save(fn)
