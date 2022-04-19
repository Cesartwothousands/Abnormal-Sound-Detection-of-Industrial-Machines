import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
import time
from Mel_specgram import *
from Mel_spectrum import *

time_start = time.time()  # time = 0
########################################################################
def CWT(path):
    y, sr = librosa.load(path, sr=None)
    t = np.linspace(0, 10, sr, endpoint=False)
    Twy, Wy, *_ = ssq_cwt(y)

    plt.figure(figsize=(10, 4))
    #plt.colorbar(format='%+2.0f dB')
    plt.title('CWT')
    plt.tight_layout()
    plt.imshow(np.abs(Wy), aspect='auto',  vmin=0, vmax=.001,cmap='turbo')
    plt.show()

def SST(path):
    y, sr = librosa.load(path, sr=None)
    t = np.linspace(0, 10, sr, endpoint=False)
    Twy, Wy, *_ = ssq_cwt(y)

    plt.figure(figsize=(10, 4))
    #plt.colorbar(format='%+2.0f dB')
    plt.title('SST')
    plt.tight_layout()
    plt.imshow(np.abs(Twy), aspect='auto', vmin=0, vmax=.001, cmap='turbo')
    plt.show()

CWT("vavle.00.normal.00000023.wav")
SST("vavle.00.normal.00000023.wav")

def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()

########################################################################
time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nTime cost', minute, 'min ', second, ' sec')