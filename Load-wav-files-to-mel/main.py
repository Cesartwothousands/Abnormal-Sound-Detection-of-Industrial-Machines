# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import numpy as np
#import os
from Mel_spectrum import mel
#from Mel_specgram import graph
from Mel_specgram import savegraph
from Pre_Data import *

time_start = time.time()  # time = 0
################################################################

for i in range(1100,1106):
    path = r'F:\毕业论文\Dataset\-6_dB_'
    picpath = r'F:\毕业论文\Pictures\梅尔频谱图\-6_dB_'
    print( i+1 , '次')

    # Load a file
    if i < num_v6n:
        path = path + r'valve\valve'
        picpath = picpath + r'valve'

        if i < num_v0a:
            path = path + r'\id_00\abnormal\\' + str(i).rjust(8, '0') +'.wav'
            picpath = picpath + r'\0\abnormal'

        elif i < num_v0n:
            path = path + r'\id_00\normal\\' + str(i - num_v0a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\0\normal'

        elif i < num_v2a:
            path = path + r'\id_02\abnormal\\' + str(i - num_v0n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\abnormal'

        elif i < num_v2n:
            path = path + r'\id_02\normal\\' + str(i - num_v2a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\normal'

        elif i < num_v4a:
            path = path + r'\id_04\abnormal\\' + str(i - num_v2n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\abnormal'

        elif i < num_v4n:
            path = path + r'\id_04\normal\\' + str(i - num_v4a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\normal'

        elif i < num_v6a:
            path = path + r'\id_06\abnormal\\' + str(i - num_v4n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\abnormal'

        else :
            path = path + r'\id_06\normal\\' + str(i - num_v6a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\normal'


    elif i < num_s6n:
        path = path + r'slider\slider'
        picpath = picpath + r'slider'

        if i < num_s0a:
            path = path + r'\id_00\abnormal\\' + str(i - num_v6n).rjust(8, '0') +'.wav'
            picpath = picpath + r'\0\abnormal'

        elif i < num_s0n:
            path = path + r'\id_00\normal\\' + str(i - num_s0a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\0\normal'

        elif i < num_s2a:
            path = path + r'\id_02\abnormal\\' + str(i - num_s0n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\abnormal'

        elif i < num_s2n:
            path = path + r'\id_02\normal\\' + str(i - num_s2a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\normal'

        elif i < num_s4a:
            path = path + r'\id_04\abnormal\\' + str(i - num_s2n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\abnormal'

        elif i < num_s4n:
            path = path + r'\id_04\normal\\' + str(i - num_s4a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\normal'

        elif i < num_s6a:
            path = path + r'\id_06\abnormal\\' + str(i - num_s4n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\abnormal'

        else :
            path = path + r'\id_06\normal\\' + str(i - num_s6a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\normal'


    elif i < num_p6n:
        path = path + r'pump\pump'
        picpath = picpath + r'pump'

        if i < num_p0a:
            path = path + r'\id_00\abnormal\\' + str(i - num_s6n).rjust(8, '0') +'.wav'
            picpath = picpath + r'\0\abnormal'

        elif i < num_p0n:
            path = path + r'\id_00\normal\\' + str(i - num_p0a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\0\normal'

        elif i < num_p2a:
            path = path + r'\id_02\abnormal\\' + str(i - num_p0n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\abnormal'

        elif i < num_p2n:
            path = path + r'\id_02\normal\\' + str(i - num_p2a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\normal'

        elif i < num_p4a:
            path = path + r'\id_04\abnormal\\' + str(i - num_p2n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\abnormal'

        elif i < num_p4n:
            path = path + r'\id_04\normal\\' + str(i - num_p4a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\normal'

        elif i < num_p6a:
            path = path + r'\id_06\abnormal\\' + str(i - num_p4n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\abnormal'

        else :
            path = path + r'\id_06\normal\\' + str(i - num_p6a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\normal'


    elif i < num_f6n:
        path = path + r'fan\fan'
        picpath = picpath + r'fan'

        if i < num_f0a:
            path = path + r'\id_00\abnormal\\' + str(i - num_p6n).rjust(8, '0') +'.wav'
            picpath = picpath + r'\0\abnormal'

        elif i < num_f0n:
            path = path + r'\id_00\normal\\' + str(i - num_f0a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\0\normal'

        elif i < num_f2a:
            path = path + r'\id_02\abnormal\\' + str(i - num_f0n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\abnormal'

        elif i < num_f2n:
            path = path + r'\id_02\normal\\' + str(i - num_f2a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\2\normal'

        elif i < num_f4a:
            path = path + r'\id_04\abnormal\\' + str(i - num_f2n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\abnormal'

        elif i < num_f4n:
            path = path + r'\id_04\normal\\' + str(i - num_f4a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\4\normal'

        elif i < num_f6a:
            path = path + r'\id_06\abnormal\\' + str(i - num_f4n).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\abnormal'

        else :
            path = path + r'\id_06\normal\\' + str(i - num_f6a).rjust(8, '0') + '.wav'
            picpath = picpath + r'\6\normal'

    I = i
    Data = mel(path)
    np.save('Data' + str(I+1), Data)

    savegraph(path, picpath)



################################################################
time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nTime cost', minute, 'min ', second, ' sec')