import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
import time

time_start = time.time()  # time = 0

def savegraph(path, filepath):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel(path), sr=16000, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')

    path = path[-12:-4]
    p = path
    p = p.rjust(8, '0')
    #    path = path.replace('.wav', '')
    fig_name = p + '.png'

    name = 'Mel spectrogram: ' + p
    plt.title(name)
    plt.tight_layout()

    f = plt.gcf()
    f.savefig(os.path.join(filepath, fig_name))  # 分别命名图片
    #    plt.show()
    f.clear()
    plt.close()

def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()

#%%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

print(xo)
print(x)
plt.plot(xo); plt.show()
plt.plot(x);  plt.show()

#%%# CWT + SSQ CWT ####################################
Twxo, Wxo, *_ = ssq_cwt(xo)
viz(xo, Twxo, Wxo)

Twx, Wx, *_ = ssq_cwt(x)
viz(x, Twx, Wx)

#%%# STFT + SSQ STFT ##################################
Tsxo, Sxo, *_ = ssq_stft(xo)
viz(xo, np.flipud(Tsxo), np.flipud(Sxo))

Tsx, Sx, *_ = ssq_stft(x)
viz(x, np.flipud(Tsx), np.flipud(Sx))



time_end = time.time()
minute = (time_end - time_start) // 60
second = (time_end - time_start) % 60
print('\nTime cost', minute, 'min ', second, ' sec')