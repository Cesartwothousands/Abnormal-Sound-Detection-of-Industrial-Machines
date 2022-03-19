import torch
from torch.utils.data import Dataset
from PIL import Image

#class MyData(Dataset):

#    def __init__(self):

#    def __getitem__(self, idx):


img_path = r"00000000.png"
img = Image.open(img_path)
