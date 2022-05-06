from Pre_Data import *
import os


path_file = r'F:\毕业论文\Pictures\梅尔频谱图'

for i in range(4):
    number = 0

    if i == 0 :
        path_filee = path_file + r'\-6_dB_valve'
    elif i == 1 :
        path_filee = path_file + r'\-6_dB_slider'
    elif i == 2 :
        path_filee = path_file + r'\-6_dB_pump'
    elif i == 3 :
        path_filee = path_file + r'\-6_dB_fan'

