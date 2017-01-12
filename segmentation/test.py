import os
import time
import cv2
import numpy as np

import tensorflow
import keras

import random

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import SGD


def build_cnn():
    model = Sequential()

    model.add(Convolution2D(7, 3, 3, border_mode='same', activation='sigmoid', input_shape=(50, 50, 1)))
    model.add(Convolution2D(3, 3, 3, border_mode='same', activation='sigmoid'))
    model.add(Convolution2D(3, 3, 3, border_mode='same', activation='sigmoid'))
    model.add(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = build_cnn()
    model.load_weights('model.md')
    image = cv2.imread('F:/dataset/image/25.bmp', flags=0)
    test_x = np.zeros((1, 50, 50, 1))
    test_x[0, :, :, 0] = image.astype(np.float32)[:50, :50]
    out = model.predict(test_x, 1)
    print(out.shape)
    print(np.max(out))
    cv2.imwrite('result.bmp', out[0, :, :, 0] * 255)
    res = cv2.resize(out[0, :, :, 0], (150, 150))
    cv2.imshow('result', res)
    cv2.waitKey(0)


