import pandas as pd
import numpy as np
import csv

A1 = np.load('pumpacc.npy')  #加载文件
A2 = np.load('pumploss.npy')  #加载文件
A3 = np.load('pumpval_acc.npy')  #加载文件
A4 = np.load('pumpval_loss.npy')  #加载文件

dataframe = pd.DataFrame({'acc':A1,'loss':A2,'vacc':A3,'vloss':A4})

dataframe.to_csv("pumpacc.csv",index=False,sep=',')