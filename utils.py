import tensorflow
import numpy as np
import os

import pandas as pd
import time

import seaborn as sns
from pylab import rcParams
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,roc_curve,auc
import matplotlib.pyplot as plt

def _get_tensorflow_version():  # pragma: no cover
    """ Utility function to decide the version of tensorflow, which will
    affect how to import keras models.
    Returns
    -------
    tensorflow version : int
    """
    tf_version = str(tensorflow.__version__)
    if int(tf_version.split(".")[0]) != 1 and int(
            tf_version.split(".")[0]) != 2:
        raise ValueError("tensorflow version error")
    return int(tf_version.split(".")[0])
# if tensorflow 2, import from tf directly

if _get_tensorflow_version() == 1:
    from keras import callbacks,regularizers
    from keras.models import Sequential
    from keras.layers import Dense, Dropout,SimpleRNN,LSTM,Activation,Masking
    from keras.regularizers import l2
    from keras.losses import mean_squared_error
else:
    from tensorflow.keras import callbacks,regularizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout,SimpleRNN,LSTM,Activation,Masking
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.losses import mean_squared_error


def mkdir(path):
    '''
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    '''
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True

#创建模型
def createDeepModel(n_cycles, n_features):
    # model = Sequential()
    # model.add(SimpleRNN(128, activation='relu', return_sequences = True, input_shape=(n_cycles, n_features)))
    # model.add(SimpleRNN(64, activation='relu', return_sequences = True))
    # model.add(SimpleRNN(32, activation='relu', return_sequences = True))
    # model.add(SimpleRNN(16, activation='relu', return_sequences = True))
    # model.add(SimpleRNN(8, activation='relu'))
    # # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1)) # <---- NOTE: not specifying an activation means that the activation is linear
    # return model

    model = Sequential()
    model.add(SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(n_cycles, n_features)))
    model.add(SimpleRNN(32, activation='relu', return_sequences=True))
    model.add(SimpleRNN(16, activation='relu', return_sequences=True))
    model.add(SimpleRNN(8, activation='relu', return_sequences=True))
    model.add(SimpleRNN(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))  # <---- NOTE: not specifying an activation means that the activation is linear
    return model

    # model = Sequential()
    # model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_cycles, n_features)))
    # model.add(Dropout(0.5))
    # model.add(LSTM(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('linear'))
    # return model
