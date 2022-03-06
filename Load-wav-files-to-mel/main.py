# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import numpy as np
from Mel_spectrum import mel
from Mel_specgram import graph
from Mel_specgram import savegraph

time_start = time.time()  # time = 0


Data = []

# Choose the type
x = 2
path = r'F:\毕业论文\Dataset\-6_dB_fan\fan\id_00\abnormal'

for i in range(0, x):

    # Load a file
    if i < 10:
        path = path + r'\0000000'+ str(i) +'.wav'
    elif i < 100:
        path = path + r'\000000'+ str(i) +'.wav'
    else:
        path = path + r'\00000'+ str(i) +'.wav'

    Data.append(mel(path))
    path = r'F:\毕业论文\Dataset\-6_dB_fan\fan\id_00\abnormal'

print(len(Data),type(Data))


################################################################
# Save data
np.save('demo.npy', Data)
################################################################
# Save picture
graph('00000000.wav')
################################################################
time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nTime cost', minute, 'min ', second, ' sec')