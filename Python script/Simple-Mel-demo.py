import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('00000000.wav', sr=None)

# 方法一：使用时间序列求Mel频谱
print(librosa.feature.melspectrogram(y=y, sr=sr))

# 方法二：使用stft频谱求Mel频谱
D = np.abs(librosa.stft(y)) ** 2  # stft频谱
S = librosa.feature.melspectrogram(S=D)  # 使用stft频谱求Mel频谱

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
