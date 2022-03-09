import librosa.display
import matplotlib.pyplot as plt
import os
from Mel_spectrum import mel

def graph(path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel(path), sr=16000, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


def savegraph(path, filepath):
    fig_name = str(path) + '.png'

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel(path), sr=16000, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    path = path.replace('.wav','')
    name = 'Mel spectrogram: ' + path
    plt.title(name)
    plt.tight_layout()

    f = plt.gcf()
    f.savefig(os.path.join(filepath , fig_name))#分别命名图片

    plt.show()
    f.clear()
    plt.close()


# graph(r'00000000.wav')
# savegraph(r'00000009.wav', r'F:\毕业论文\Pictures\梅尔频谱图')