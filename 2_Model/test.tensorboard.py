from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

img_path = r"00000000.png"
img_PTL = Image.open(img_path)
img_array = np.array(img_PTL)
print(img_array.shape)
writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x^2", i*i ,i)

writer.close()