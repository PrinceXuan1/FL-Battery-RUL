import keras
import numpy as np
import pandas as pd
import time
import datetime
import os
# import psutil
import random
from sklearn.model_selection import train_test_split
import csv
from utils import mkdir,createDeepModel, _get_tensorflow_version
import tensorflow
from keras.optimizers import SGD, Adam

numOfIterations=100
numOfClients=5
mergebatterynum=24  #和step1保持一致

#下面这些不用调
n_cycles = 100
n_features=17
clientsModelList=[]
deepModelAggWeights=[]
firstClientFlag=True
datapath='./FLdataset/'
testdatapath='./FLdataset/test/'
modelLocation='./FLmodel/FL_RNN_model.h5'


def updateServerModel(deepModelAggWeights,clientModelWeight):
    global firstClientFlag
    for ind in range(len(clientModelWeight)):
        if(firstClientFlag==True):
            deepModelAggWeights.append(clientModelWeight[ind])
        else:
            deepModelAggWeights[ind]=(deepModelAggWeights[ind]+clientModelWeight[ind])

def updateClientsModels():
    global clientsModelList
    global deepModel
    clientsModelList.clear()
    for clientID in range(numOfClients):
        m = keras.models.clone_model(deepModel)
        m.set_weights(deepModel.get_weights())
        clientsModelList.append(m)

# Run the main Method
if __name__ == '__main__':
    #step 0 加载测试集
    testdata=pd.read_csv(testdatapath+'01_Battery_Feature.csv',header=None)
    testlabel=pd.read_csv(testdatapath+'02_Battery_RUL.csv',header=None)
    testdata=testdata.values.reshape((61*3,n_cycles,n_features))
    testlabel=testlabel.values.reshape(testlabel.shape[0])
    print("testdata",testdata.shape)
    print("testlabel",testlabel.shape)

    #step1 创建初始模型
    deepModel=createDeepModel(n_cycles, n_features)
    for clientID in range(numOfClients):
        clientsModelList.append(deepModel)
    #开始联邦训练
    for iterationNo in range(numOfIterations):
        print("iterationNo", iterationNo)
        #开始本地训练
        for clientID in range(numOfClients):
            #step2 读取数据集
            print("clientID",clientID)
            # trainNum=random.randint(0, 3)
            clientdatapath=datapath+str(clientID)+'/01_Battery_Feature.csv'
            clientlabelpath=datapath+str(clientID)+'/02_Battery_RUL.csv'
            clientdata=pd.read_csv(clientdatapath,header=None)
            clientlabel=pd.read_csv(clientlabelpath,header=None)
            clientdata=clientdata.values.reshape((61*mergebatterynum,n_cycles,n_features))
            clientlabel=clientlabel.values.reshape(clientlabel.shape[0])

            #step3 开始本地训练  Adam(learning_rate=0.0005, clipnorm = 0.001)
            clientsModelList[clientID].compile(optimizer=Adam(lr=0.0005, clipnorm = 0.01), loss='mse', metrics=['mae'])
            clientsModelList[clientID].fit(clientdata, clientlabel, epochs=10, verbose=1, batch_size =64, validation_data=(testdata, testlabel))
            clientWeight=clientsModelList[clientID].get_weights()
            updateServerModel(deepModelAggWeights,clientWeight)

            clientsModelList[clientID].save("./FLmodel/FL_RNN_model_node_"+str(clientID)+".h5")
            firstClientFlag=False

        #step4 模型全局聚合
        print("deepModelAggWeights",len(deepModelAggWeights))
        for ind in range(len(deepModelAggWeights)):
            deepModelAggWeights[ind]/=numOfClients
        dw_last=deepModel.get_weights()

        for ind in range(len(deepModelAggWeights)):
            dw_last[ind]=deepModelAggWeights[ind]

        #Update server's model
        deepModel.set_weights(dw_last)
        print("Server's model updated")
        print("Saving model . . .")
        deepModel.save(modelLocation)
        #step5 更新子节点权重
        updateClientsModels()
        firstClientFlag=True
        deepModelAggWeights.clear()


