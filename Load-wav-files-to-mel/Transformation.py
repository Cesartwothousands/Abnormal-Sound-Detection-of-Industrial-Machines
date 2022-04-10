import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

time_start = time.time()  # time = 0

y, sr = librosa.load('vavle.00.normal.00000023.wav', sr=None)
print("y =",y,"\n","sr =",sr)

def timedomain(i):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.subplot(311)
    plt.plot(i)
    plt.xlabel('sample')
    plt.ylabel('amplitude')
    f = plt.gcf()
    f.savefig(os.path.join("F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel", "Timedomain"))
    plt.close()

def frequencydomain(i):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.subplot(311)
    plt.plot(cent[0])
    plt.xlabel('sample')
    plt.ylabel('Frequency')
    f = plt.gcf()
    f.savefig(os.path.join("F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel", "frequencydomain"))
    plt.close()

timedomain(y)
frequencydomain(y)

time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nTime cost', minute, 'min ', second, ' sec')