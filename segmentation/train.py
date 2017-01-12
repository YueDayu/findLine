import os
import time
import cv2
import numpy as np

import tensorflow as tf
import keras

import random

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.optimizers import SGD
from keras.optimizers import Adam
import keras.backend as K
from keras.backend.common import _EPSILON

base_path = 'F:/dataset/'


def load_data():
    image_path = os.path.join(base_path, 'image')
    label_path = os.path.join(base_path, 'label')
    all_filename = os.listdir(image_path)
    random.shuffle(all_filename)

    train_image = [os.path.join(image_path, x) for x in all_filename[:425]]
    train_label = [os.path.join(label_path, x) for x in all_filename[:425]]
    test_image = [os.path.join(image_path, x) for x in all_filename[425:]]
    test_label = [os.path.join(label_path, x) for x in all_filename[425:]]

    train_x = np.zeros((len(train_image), 50, 50, 1))
    train_y = np.zeros((len(train_label), 50, 50, 1))
    for (i, x) in enumerate(train_image):
        train_x[i, :, :, 0] = cv2.imread(x, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32)[:50, :50]
    for (i, x) in enumerate(train_label):
        image = cv2.imread(x, flags=cv2.IMREAD_GRAYSCALE)[:50, :50]
        image[image > 0] = 1
        train_y[i, :, :, 0] = image

    test_x = np.zeros((len(test_image), 50, 50, 1))
    test_y = np.zeros((len(test_image), 50, 50, 1))
    for (i, x) in enumerate(test_image):
        test_x[i, :, :, 0] = cv2.imread(x, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32)[:50, :50]
    for (i, x) in enumerate(test_label):
        image = cv2.imread(x, flags=cv2.IMREAD_GRAYSCALE)[:50, :50]
        image[image > 0] = 1
        test_y[i, :, :, 0] = image

    return train_x, train_y, test_x, test_y


def build_cnn():
    model = Sequential()

    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='sigmoid', input_shape=(50, 50, 1)))
    model.add(Convolution2D(4, 3, 3, border_mode='same', activation='sigmoid'))
    model.add(Convolution2D(2, 3, 3, border_mode='same', activation='sigmoid'))
    model.add(Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid'))

    return model


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def binary_crossentropy(output, target, from_logits=False):
    if not from_logits:
        epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.log(output / (1 - output))
    try:
        return tf.nn.weighted_cross_entropy_with_logits(output, target, 100)
    except TypeError:
        return tf.nn.weighted_cross_entropy_with_logits(output, target, 100)


def weighted_loss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_pred, y_true), axis=-1)


def main():
    print('loading data')
    (train_x, train_y, test_x, test_y) = load_data()
    print('load data complete')

    model = build_cnn()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.01, decay=1e-6)
    model.compile(loss=weighted_loss, optimizer=adam, metrics=['accuracy'])
    print('build model complete')

    print('start training')
    model.fit(train_x, train_y, batch_size=100, nb_epoch=1000,
              verbose=2,
              shuffle=True,
              validation_data=(test_x, test_y))
    model.save_weights('model.md')


if __name__ == '__main__':
    main()

