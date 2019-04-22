# simple regression as described in video
import sys
import numpy as np
import cv2
import csv

if len(sys.argv) < 2:
    print('usage:')
    print('   python {} <training-set-name>'.format(sys.argv[0]))
    quit()

tsname = str(sys.argv[1])

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

X = np.array(limg)
del limg
Y = np.array(langle)
del langle

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

print('[info] creating and compiling model')
model.compile(loss='mse',optimizer='adam')

print('[info] !! fitting')
model.fit(X,Y,validation_split=0.2,shuffle=False,epochs=30)

fn = '{}.model'.format(tsname)
print('[info] saving model to "{}"'.format(fn))
model.save(fn)
