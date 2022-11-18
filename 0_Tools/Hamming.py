import os
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import numpy as np

window = np.hamming(100)

plt.figure(figsize=(6, 4))
plt.subplots_adjust(wspace=1, hspace=0.5)
#    plt.subplot(311)
plt.plot(window)
plt.xlabel('Sample')
plt.ylabel('Amplitude')
f = plt.gcf()
f.savefig(os.path.join("F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel",
                           "Hamming window"))
plt.close()