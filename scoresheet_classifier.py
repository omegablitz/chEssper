#!/usr/bin/python

import numpy as np
from keras import models
from matplotlib import pyplot as plt
import cv2
import sys

ims = []
for im in sys.argv[1:]:
    im = cv2.imread(im, 0).tolist()
    im = np.array([[[[x] for x in y] for y in im]]).T
    im = im.astype('float32')
    im /= 255
    ims.append(im[0])
ims = np.array(ims)

model = models.load_model('./cnn1.h5')

mappings_x = open('binary/emnist-balanced-mapping.txt').readlines()
mappings_x = map(lambda x: x.strip(), mappings_x)
mappings = {}
for line in mappings_x:
    spl = line.split(' ')
    mappings[int(spl[0])] = int(spl[1])


# print np.amax(model.predict_proba(ims), axis=1)
preds = []
for title, prediction in zip(sys.argv[1:], model.predict_classes(ims)):
    prediction = unichr(mappings[prediction])
    split_title = title.split('_')
    move_number = split_title[0].split('/')[-1]
    char_idx = split_title[1].split('.')[0]
    preds.append(((int(move_number), int(char_idx)), prediction))
preds.sort(key=lambda x: (x[0][0], x[0][1]))


class Move():
    def __init__(self, move):
        self.move = move
    def get_move(self):
        if len(self.move) == 0:
            return None
        move = self.move.replace('X', 'x').replace('C', 'c')
        
        if move[0] == '6':
            move = 'b' + move[1:]
        elif move[0] == '8':
            move = 'B' + move[1:]

        if move[1] == 'K':
            move = move[0] + 'x' + move[2:]

        if move[-1] == 'K' or move[-1] == 'f':
            move = move[:-1] + '+'
            if move[-2] == 'b':
                move = move[:-2] + '6+'
            if move[-3] == '6':
                move = move[:-3] + 'b' + move[-2:]
        else:
            if move[-1] == 'b':
                move = move[:-1] + '6'
            if move[-2] == '6':
                move = move[:-2] + 'b' + move[-1]
        
        if move[-1] == '4' and move[-2] in '12345678' and move[-3] in 'abcdefgh':
            move = move[:-1] + '#'


        if move.count('O') == 3:
            move = 'O-O-O'
        elif move.count('O') == 2:
            x = 'O-O'
        return move

current = None
so_far = ''
moves = []
for (move_number, char_idx), pred in preds:
    if (move_number, char_idx / 5) != current:
        current = (move_number, char_idx / 5)
        split = so_far.split(' ')
        for _ in split:
            moves.append(Move(_))
        so_far = ''
    so_far += pred
if len(so_far) > 0:
    moves.append(Move(so_far))

print '<MOVES>'
moves = filter(lambda x: x.get_move() != None, moves)
for idx, move in enumerate(moves):
    if idx % 2 == 0:
        buf = str(idx / 2 + 1) + '. ' + str(move.get_move())
    else:
        print buf + ' ' + str(move.get_move())
        buf = None
if buf != None:
    print buf
