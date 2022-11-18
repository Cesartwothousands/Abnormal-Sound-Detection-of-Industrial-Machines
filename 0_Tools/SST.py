import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
import time
from Mel_specgram import *
from Mel_spectrum import *
import math

time_start = time.time()  # time = 0
########################################################################
def CWT(path):
    y, sr = librosa.load(path, sr=None)
    t = np.linspace(0, 10, sr, endpoint=False)
    Twy, Wy, *_ = ssq_cwt(y)

    plt.figure(figsize=(6, 4))
    #plt.colorbar(format='%+2.0f dB')
    plt.title('CWT')
    plt.tight_layout()
    plt.imshow(np.abs(Wy), aspect='auto' , vmin=0, vmax=.05,cmap='turbo')
    plt.show()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    f = plt.gcf()
    f.savefig(os.path.join("F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel",
                           "23CWT"))
    plt.close()

def SST(path):
    y, sr = librosa.load(path, sr=None)
    t = np.linspace(0, 10, sr, endpoint=False)
    Twy, Wy, *_ = ssq_cwt(y)

    plt.figure(figsize=(6, 4))
    #plt.colorbar(format='%+2.0f dB')
    plt.title('SST')
    plt.tight_layout()
    plt.imshow(np.abs(Twy), aspect='auto', vmin=0, vmax=.001, cmap='turbo')
    plt.show()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    f = plt.gcf()
    f.savefig(os.path.join("F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel",
                           "23SST"))
    plt.close()

def MelSST(path):
    y, sr = librosa.load(path, sr=None)
    t = np.linspace(0, 10, sr, endpoint=False)
    Twy, Wy, *_ = ssq_cwt(y)
    Twy = Wy
    for i in range(len(Twy)):
        for j in range(len(Twy)):
            Twy[i][j] = 1127 * math.log(np.abs(Twy[i][j])/700 + 1)

    plt.figure(figsize=(10, 4))
    #plt.colorbar(format='%+2.0f dB')
    plt.title('SST')
    plt.tight_layout()
    plt.imshow(np.abs(Twy), aspect='auto', cmap='turbo')
    plt.show()

CWT("vavle.00.normal.00000023.wav")
SST("vavle.00.normal.00000023.wav")
# MelSST("vavle.00.normal.00000023.wav")

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