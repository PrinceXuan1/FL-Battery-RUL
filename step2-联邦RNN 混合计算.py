import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as kb
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
import seaborn as sns
sns.set()

numOfIterations=10
numOfClients=5

#下面这些不用调
# Number of cycles to use for prediction
n_cycles = 100

# number of features
n_features = 17
clientsModelList=[]
deepModelAggWeights=[]
firstClientFlag=True
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

##########################################数据预处理#################################################
dfs = pd.read_hdf('./origindata/battery_summary_data_features.h5')
drop_columns = ['RUL', 'chargetime']
# Let's try to clean the data, since there are a lot of spikes that probably
# represent some error in the sensor reading.
# We employ Exponential Moving Averages in order to "smoothen out" the problematic curves.
ir = []
qcharge = []
qdischarge = []
dq = []
for i in dfs.index.levels[0]:
    ewm_ir = dfs.loc[(i, slice(None)),'IR'].ewm(span=5,adjust=False).mean().values
    ewm_qc = dfs.loc[(i, slice(None)),'QCharge'].ewm(span=5,adjust=False).mean().values
    ewm_qd = dfs.loc[(i, slice(None)),'QDischarge'].ewm(span=5,adjust=False).mean().values
    ewm_dq = dfs.loc[(i, slice(None)),'DQ'].ewm(span=5,adjust=False).mean().values
    ir = np.concatenate((ir, ewm_ir), axis=None)
    qcharge = np.concatenate((qcharge, ewm_qc), axis=None)
    qdischarge = np.concatenate((qdischarge, ewm_qd), axis=None)
    dq = np.concatenate((dq, ewm_dq), axis=None)
dfs['IR'] = ir
dfs['QCharge'] = qcharge
dfs['QDischarge'] = qdischarge
dfs['DQ'] = dq

# The idea is to use a RNN to predict what happens at later cycles.
# Get 60 "windows" of 100 cycles from the first 160 cycles as your whole data.
# Treat this as a number of time-series, each with n_features features (all except RUL)
# The data then needs to be organized as such: (batch_size, timesteps, features)
# and needs to maintain the time-ordering of the sequence.
X = []
y = []
for i in dfs.index.levels[0]:
    for j in range(61):
        X.append(dfs.loc[(i, slice(j+1,n_cycles+j)), :].drop(columns=drop_columns).values)
        y.append(dfs.loc[(i, n_cycles+j), :].RUL)
X = np.array(X)
y = np.array(y)
print('Input array shape: ', X.shape)

# Split in train/test sets. The data is also randomply selected
# and this is important, since the daa is ordered - by construction.
from sklearn.model_selection import train_test_split
xClients,X_test,yClients,y_test = train_test_split(X,y,test_size=0.1)
print('batch size, timesteps, features')
print('xClients', xClients.shape)
print('yClients', yClients.shape)
print('X_test ', X_test.shape)
print('y_test ', y_test.shape)
n_train_samples = xClients.shape[0]

# number of test entries
n_test_samples = X.shape[0]-xClients.shape[0]
print("n_train_samples",n_train_samples)
print("n_test_samples",n_test_samples)

# Scale the data
xClients = xClients.reshape(n_train_samples*n_cycles,n_features)
X_test = X_test.reshape(n_test_samples*n_cycles,n_features)
print("xClients",xClients.shape)
print("X_test_scale",X_test.shape)
# scaler1 = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(xClients)
xClients = scaler.transform(xClients)
X_test = scaler.transform(X_test)
xClients = xClients.reshape(n_train_samples,n_cycles,n_features)
X_test = X_test.reshape(n_test_samples,n_cycles,n_features)


#step1 创建初始模型
deepModel=createDeepModel(n_cycles, n_features)
for clientID in range(numOfClients):
    clientsModelList.append(deepModel)

xClientsList=[]
yClientsList=[]

clientsModelList=[]
historyList=[[],[],[],[],[]]
clientDataInterval=len(xClients)//numOfClients


lastLowerBound=0
for clientID in range(numOfClients):
    xClientsList.append(xClients[lastLowerBound : lastLowerBound+clientDataInterval])
    yClientsList.append(yClients[lastLowerBound : lastLowerBound+clientDataInterval])
    clientsModelList.append(deepModel)
    lastLowerBound+=clientDataInterval

for iterationNo in range(1,numOfIterations+1):
    for clientID in range(numOfClients):
        clientsModelList[clientID].compile(optimizer=Adam(lr=0.0005,clipnorm = 0.001), loss='mse', metrics=['mae'])
        print('xClientsList[clientID]',xClientsList[clientID].shape)
        history = clientsModelList[clientID].fit(xClientsList[clientID], yClientsList[clientID], epochs=1, verbose=1, batch_size =32, validation_data=(X_test, y_test))
        historyList[clientID].append(history)

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


# Graph of the loss function for both train and validation sets
for clientID in range(numOfClients):
    df_list = []
    for i in range(len(historyList[clientID])):
        df = pd.DataFrame(historyList[clientID][i].history['loss'], columns = ['loss'])
        df['val_loss'] = historyList[clientID][i].history['val_loss']
        df_list.append(df)
    loss = pd.concat(df_list, axis=0, ignore_index=True)
    print(loss)
    ax = sns.lineplot(data=loss)
    title = 'client' + str(clientID)
    ax.set(xlabel='Epochs', ylabel='MSE loss', title=title)
    plt.show()

# Graph of the MAE for both train and validation sets
for clientID in range(numOfClients):
    df_list = []
    for i in range(len(historyList[clientID])):
        df = pd.DataFrame(historyList[clientID][i].history['mean_absolute_error'], columns = ['MAE'])
        df['val_MAE'] = historyList[clientID][i].history['val_mean_absolute_error']
        df_list.append(df)
    loss = pd.concat(df_list, axis=0, ignore_index=True)
    ax = sns.lineplot(data=loss)
    title = 'client' + str(clientID)
    ax.set(xlabel='Epochs', ylabel='MAE', title=title)
    plt.show()


scores_train = deepModel.evaluate(xClients, yClients)
scores_test = deepModel.evaluate(X_test, y_test)
print('MSE loss, MAE train:', scores_train)
print('MSE loss, MAE test:', scores_test)

# Graph of the RUL predictions
outputs = deepModel.predict(X_test)
sns.set(rc={'figure.figsize':(18,6)})
df = pd.DataFrame(outputs[0:100], columns = ['predictions'])
df['targets'] = y_test[0:100]
ax = sns.lineplot(data=df)
ax.set(xlabel='# of test samples', ylabel='RUL')
plt.show()

# calculate residuals
residuals = [y_test[i]-outputs[i] for i in range(len(outputs))]
residuals = pd.DataFrame(residuals)
ax = sns.lineplot(data=residuals)
ax.set(xlabel='# of test samples', ylabel='Residual')
plt.show()

# summary statistics for residuals
print(residuals.describe())

# Plot residuals: check if they are normal-like shaped
sns.set(rc={'figure.figsize':(6,6)})
sns.distplot(residuals, bins=50, kde=True)
plt.show()