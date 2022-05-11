# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import librosa
import matplotlib.pyplot as plt

def mel(path):

    # Load a file
    y, sr = librosa.load(path, sr=None)

    # Extract mel spectrogram
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    # 128表示Mel频率的维度（频域），194为时间帧长度（时域），
    # 所以Log-Mel Spectrogram特征是音频信号的时频表示特征。
    # 其中，n_fft指的是窗的大小，这里为1024
    # hop_length表示相邻窗之间的距离，这里为512，也就是相邻窗之间有50%的overlap；
    # n_mels为mel bands的数量，这里设为128。

    # Convert to log scale
    logmelspec = librosa.power_to_db(melspec)

    return logmelspec

def stft(path):
    # Load a file
    data = lc.load(path,sr=None)

    length = len(data[0])

    spec = np.array(lc.stft(data[0], n_fft=512, hop_length=160, win_length=400, window='hann'))

    plt.pcolormesh(np.array(range(int(length/160+1)))/fs, f, np.abs(spec))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


stft('vavle.00.normal.00000023.wav')
#a = mel(r'00000000.wav')
#print(type(a))
# mel('F:\\毕业论文\\Dataset\\-6_dB_slider\\slider\\id_00\\abnormal\\\\00000001.wav')