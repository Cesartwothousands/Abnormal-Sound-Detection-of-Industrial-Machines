import pandas as pd
import numpy as np
import csv

A1 = np.load('acc.npy')  #加载文件
A2 = np.load('loss.npy')  #加载文件
A3 = np.load('val_acc.npy')  #加载文件

dataframe = pd.DataFrame({'acc':A1,'loss':A2,'vacc':A3})

dataframe.to_csv("acc.csv",index=False,sep=',')