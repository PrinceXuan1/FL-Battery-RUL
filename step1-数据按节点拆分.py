import numpy as np
import pandas as pd
import time
import datetime
import os
# import psutil
import random
from sklearn.model_selection import train_test_split
import csv
from utils import mkdir
from sklearn.preprocessing import MinMaxScaler

#合并电池的数量
mergebatterynum=24
nodenum=5  #必须小于 120/mergebatterynum  120个做训练集 最后3个做测试集
origindatapath='./origindata/origindata.csv'
drop_columns = ['RUL', 'chargetime']


#合并电池数据
def mergedata(origindatapath,mergebatterynum,nodenum):
    datanum=0
    #初始化归一化
    origindata= pd.read_csv(origindatapath).drop(columns=drop_columns).values
    print(origindata)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #计算出max - min
    scaler.fit(origindata)
    #数据按节点进行合并
    for node in range(nodenum):
        datapath='./dataset/'
        fldatapath='./FLdataset/'
        print("node",node)
        data_list = []
        label_list= []
        #将几个数据集进行合并
        for i in range(mergebatterynum):
            datanum=datanum+1
            print("datanum",datanum)
            datafilepath=datapath+str(i)+'/01_Battery_Feature.csv'
            labelfilepath=datapath+str(i)+'/02_Battery_RUL.csv'
            data = pd.read_csv(datafilepath,header=None)
            label = pd.read_csv(labelfilepath,header=None)
            #数据归一化  归一化 到 -1~1的范围
            data = scaler.transform(data)
            data=pd.DataFrame(data)
            data_list.append(data)
            label_list.append(label)
        filedata = pd.concat(data_list,axis=0)
        labeldata= pd.concat(label_list,axis=0)
        print("filedata",filedata.shape)
        print("labeldata",labeldata.shape)
        flfilepath=fldatapath+str(node)+'/'
        mkdir(flfilepath)
        filedata.to_csv(flfilepath+'01_Battery_Feature.csv',index=False,sep=',',header=False)#保存为CSV文件
        labeldata.to_csv(flfilepath+'02_Battery_RUL.csv',index=False,sep=',',header=False)#保存为CSV文件

    #加载测试集
    # data = pd.read_csv('./dataset/123/01_Battery_Feature.csv',header=None)
    # label = pd.read_csv('./dataset/123/02_Battery_RUL.csv',header=None)
    # data = scaler.transform(data)
    # data=pd.DataFrame(data)
    # testdatapath='./FLdataset/test/'
    # mkdir(testdatapath)
    # data.to_csv(testdatapath+'01_Battery_Feature.csv',index=False,sep=',',header=False)#保存为CSV文件
    # label.to_csv(testdatapath+'02_Battery_RUL.csv',index=False,sep=',',header=False)#保存为CSV文件
    datapath = './dataset/'
    fldatapath = './FLdataset/'
    data_list1 = []
    label_list1 = []
    for i in range(3):
        datanum=datanum+1
        print("datanum",datanum)
        datafilepath=datapath+str(i+121)+'/01_Battery_Feature.csv'
        labelfilepath=datapath+str(i+121)+'/02_Battery_RUL.csv'
        data = pd.read_csv(datafilepath,header=None)
        label = pd.read_csv(labelfilepath,header=None)
        #数据归一化  归一化 到 -1~1的范围
        data = scaler.transform(data)
        data=pd.DataFrame(data)
        data_list1.append(data)
        label_list1.append(label)
    filedata = pd.concat(data_list1,axis=0)
    labeldata= pd.concat(label_list1,axis=0)
    print("filedata",filedata.shape)
    print("labeldata",labeldata.shape)
    testdatapath = './FLdataset/test/'
    mkdir(testdatapath)
    filedata.to_csv(testdatapath + '01_Battery_Feature.csv', index=False, sep=',', header=False)  # 保存为CSV文件
    labeldata.to_csv(testdatapath+'02_Battery_RUL.csv',index=False,sep=',',header=False)#保存为CSV文件


# Run the main Method
if __name__ == '__main__':
    if 120/mergebatterynum < nodenum:
        print("电池数据总量不够节点分，请减小合并数量或节点数量！")
    else:
        mergedata(origindatapath,mergebatterynum,nodenum)
