#!/usr/bin/python

import numpy as np
import cv2
import sys
ims = []
for im in sys.argv[1:]:
    # im = cv2.imread("Pictures/N-28-resized.png", 0).tolist()
    im = cv2.imread(im, 0).tolist()
    im = np.array([[[[x] for x in y] for y in im]]).T
    # print im.shape
    ###############im = 255 - im
    im = im.astype('float32')

    im /= 255
    ims.append(im[0])
    # print im
    # print im.shape
ims = np.array(ims)

from keras import models
model = models.load_model('./cnn1.h5')

mappings_x = open('binary/emnist-balanced-mapping.txt').readlines()
mappings_x = map(lambda x: x.strip(), mappings_x)
mappings = {}
for line in mappings_x:
    spl = line.split(' ')
    mappings[int(spl[0])] = int(spl[1])



#print 'label', unichr(mappings[model.predict_classes([im])[0]])
from matplotlib import pyplot as plt
idx = 0
correct = 0
correct_dict = {}
total_dict = {}
print np.amax(model.predict_proba(ims), axis=1)
preds = []
for title, prediction in zip(sys.argv[1:], model.predict_classes(ims)):
    prediction = unichr(mappings[prediction])
    preds.append((title, prediction))
    if prediction not in total_dict:
        total_dict[prediction] = 0
        correct_dict[prediction] = 0
    # if title.split('/')[-1][0] == prediction:
    #     correct += 1
    #     correct_dict[prediction] += 1
    #     #else:
    #print preds[-1]
    #a = np.array([[round(y[0]*255) for y in x] for x in ims[idx]]).T.tolist()
    #plt.imshow(a, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    #plt.show()
    total_dict[prediction] += 1
    idx += 1
preds.sort(key=lambda x: (int(x[0].split('_')[0].split('/')[-1]), int(x[0].split('_')[1].split('.')[0])))
current = None
flush = []
idx = 0
for move, pred in preds:
    if move.split('/')[-1].split('_')[0] != current:
        current = move.split('/')[-1].split('_')[0]
        if flush and flush[-1] == 'f':
            flush[-1] = '+'
        out = ''.join(flush)
        split = out.split(' ')
        out = []
        for x in split:
            if x.count('O') == 3:
                x = 'O-O-O'
            if x.count('O') == 2:
                x = 'O-O'
            out.append(x)
        print ' '.join(out)
        flush = []
    if int(move.split('_')[1].split('.')[0]) == 5:
        if flush and flush[-1] == 'f':
            flush[-1] = '+'
        flush.append(' ')
        idx = 0
    if pred == 'X':
        pred = 'x'
    elif pred == 'C':
        pred = 'c'
    if pred == 'K' and idx > 0:
        pred = 'x'
    flush.append(pred)
    idx += 1
print ''.join(flush)

    #a = np.array([[round(y[0]*255) for y in x] for x in ims[idx]]).T.tolist()

    #plt.imshow(a, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    #plt.show()
# print 'label', unichr(mappings[model.predict_classes(ims)[0]])

# print '# correct', correct
# print 'out of', idx
# print correct_dict
# print total_dict
