import pandas as pd

data = pd.read_csv('./FLdataset/0/02_Battery_RUL.csv',header=None)
print(data.values.reshape(data.shape[0]).shape)