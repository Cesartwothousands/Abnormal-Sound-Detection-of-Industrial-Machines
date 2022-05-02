from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = r"00000000.png"
image = Image.open(img_path)

writer = SummaryWriter("logs" )

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image)

writer.add_image("Tensor_img", tensor_img)

print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5,0.5], [0.5,0.5,0.5,0.5])
tensor_img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
writer.add_image("Tensor_img_normalize", tensor_img_norm)


writer.close()