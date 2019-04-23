# 1st version of CNN; compute loss on test set
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
fn = '{}_test.csv'.format(tsname)
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

X = np.array(limg)
del limg
Y = np.array(langle)
del langle

from keras.models import load_model

fn = '{}.cnn.model2'.format(tsname)
model = load_model(fn)

r = model.evaluate(X,Y)
print('r:',type(r),r)
