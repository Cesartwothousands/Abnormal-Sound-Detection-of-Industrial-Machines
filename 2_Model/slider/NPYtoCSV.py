import pandas as pd
import numpy as np
import csv

A1 = np.load('slideracc.npy')  #加载文件
A2 = np.load('sliderloss.npy')  #加载文件
A3 = np.load('sliderval_acc.npy')  #加载文件
A4 = np.load('sliderval_loss.npy')  #加载文件

dataframe = pd.DataFrame({'acc':A1,'loss':A2,'vacc':A3,'vloss':A4})

dataframe.to_csv("slideracc.csv",index=False,sep=',')