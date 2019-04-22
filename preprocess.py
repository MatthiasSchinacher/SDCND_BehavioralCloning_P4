import sys
import numpy as np
import csv

if len(sys.argv) < 3:
    print('usage:')
    print('   python {} <training-set-name> <input-files> ...'.format(sys.argv[0]))
    quit()

tsname = str(sys.argv[1])

all_lines = []
for i in range(2,len(sys.argv)):
    with open(sys.argv[i]) as csvfile:
        print('[info] reading {}'.format(sys.argv[i]))
        reader = csv.reader(csvfile)
        for l in reader:
            all_lines.append(l)

l_all = len(all_lines)
l_train = int(0.7*float(l_all))
l_valid = int(0.2*float(l_all))
l_test = l_all - l_train - l_valid
print('[info] l_test:',l_test,'; l_train:',l_train,'; l_valid:',l_valid)

train_lines = []
high = l_all
for i in range(0,l_train):
    idx = int(np.random.randint(0,high=high))
    high -= 1
    train_lines.append(all_lines.pop(idx))

valid_lines = []
for i in range(0,l_valid):
    idx = int(np.random.randint(0,high=high))
    high -= 1
    valid_lines.append(all_lines.pop(idx))

test_lines = []
for i in range(0,l_test):
    idx = int(np.random.randint(0,high=high))
    high -= 1
    test_lines.append(all_lines.pop(idx))

fn = '{}_train.csv'.format(tsname)
with open(fn, 'w') as f:
    print('[info] writing {}'.format(fn))
    writer = csv.writer(f)
    writer.writerows(train_lines)

fn = '{}_valid.csv'.format(tsname)
with open(fn, 'w') as f:
    print('[info] writing {}'.format(fn))
    writer = csv.writer(f)
    writer.writerows(valid_lines)

fn = '{}_test.csv'.format(tsname)
with open(fn, 'w') as f:
    print('[info] writing {}'.format(fn))
    writer = csv.writer(f)
    writer.writerows(test_lines)
