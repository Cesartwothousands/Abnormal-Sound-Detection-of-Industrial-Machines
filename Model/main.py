import torch
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]
)

data_path1 = r'F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Model\train'
data_path2 = r'F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Model\test'
# 获取数据集
# 数据处理


transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5]) # 标准化至[-1,1]
])


#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataSet=FlameSet(r'F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Model\train')
    print(dataSet[0])


# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x
model = CNNnet()
# print(model)

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

loss_count = []
for epoch in range(2):
    for i,(x,y) in enumerate(dataSet):
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        batch_y = Variable(y) # torch.Size([128])
        # 获取最后输出
        out = model(batch_x) # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out,batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i%20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
            torch.save(model,r'C:\Users\liev\Desktop\myproject\yin_test\log_CNN')
        if i % 100 == 0:
            for a,b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = model(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                print('accuracy:\t',accuracy.mean())
                break
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count,label='Loss')
plt.legend()
plt.show()

