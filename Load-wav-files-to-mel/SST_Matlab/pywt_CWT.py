import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.font_manager import FontProperties

sampling_rate = 16000
t = np.arange(0, 10, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300

path = "vavle.00.normal.00000023.wav"
y, sr = librosa.load(path, sr=None)

data = y

wavename = 'cgau8'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"时间(秒)")
plt.title(u"300Hz和200Hz和100Hz的分段波形和时频谱", fontsize=20)
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"频率(Hz)")
plt.xlabel(u"时间(秒)")
plt.subplots_adjust(hspace=0.4)
plt.show()