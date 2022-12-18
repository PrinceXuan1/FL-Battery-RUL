import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
n_cycles = 100
drop_columns=['chargetime','RUL']
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

dfs = pd.read_hdf('./origindata/battery_summary_data_features.h5')
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

totalcount=0
idx=0
for i in dfs.index.levels[0]:
    rullist=[]
    path='./dataset/'+str(idx)+'/'
    mkdir(path)
    for j in range(61):
        data=dfs.loc[(i, slice(j+1,n_cycles+j)), :].drop(columns=drop_columns).values
        rul=dfs.loc[(i, n_cycles+j), :].RUL
        rullist.append(rul)
        data = pd.DataFrame(data)
        data.to_csv(path+'01_Battery_Feature.csv',encoding='utf-8', sep=',', index=False,mode='a',header=False)
    rullist = np.array(rullist)
    rullist = pd.DataFrame(rullist)
    rullist.to_csv(path+'02_Battery_RUL.csv',encoding='utf-8', sep=',', index=False,header=False)
    idx=idx+1





