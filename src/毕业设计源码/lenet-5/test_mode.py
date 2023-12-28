# coding=utf-8
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
(X_train, y_train), (X_test, y_test) = mnist.load_data()


def one_pic_test():
    lena = mpimg.imread('./a.bmp')  # 二值图 读取和代码处于同一目录下的
    plt.imshow(lena)
    # plt.show()  # 显示图片
    lena = lena.reshape(1, 28, 28, 1).astype('float32')
    lena = lena / 255  # 统一格式
    model = load_model('./lenet_5.h5')
    pre = model.predict_classes(lena)  # 预测
    print(pre)


one_pic_test()
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(X_train[i], cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i]))
#     plt.show()
# plot 4 images as gray scale
# plt.subplot(221)
# plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
# # show the plot
# plt.show()
# data for tensorflow
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)
