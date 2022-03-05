# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import librosa
import librosa.display


def mel(path):
    F = []

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
    F.append(logmelspec)

    return F
