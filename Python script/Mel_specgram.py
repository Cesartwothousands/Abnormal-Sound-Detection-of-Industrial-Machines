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

# graph(r'00000000.wav')
# savegraph(r'00000009.wav', r'F:\毕业论文\Pictures\梅尔频谱图')
# savegraph('F:\\毕业论文\\Dataset\\-6_dB_slider\\slider\\id_00\\abnormal\\\\00000001.wav','F:\\毕业论文\\Pictures\\梅尔频谱图\\-6_dB_slider\\0\\abnormal')
