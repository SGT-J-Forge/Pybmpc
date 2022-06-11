import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import preprocess
import crypten.mpc as mpc
import crypten.communicator as comm
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
import Pybmpc

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh(inplace=True)
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh(inplace=True)
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return x

model = Net()

model = torch.load('./results/MLP.pt')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=False)

print('--------------------Begin-------------------------')
for i, (input, label) in enumerate(train_loader):
    with torch.no_grad():
        result = Pybmpc.ourCompute(input, model)
        print("Our MPC result:", result.argmax(1))
        print("label:", label)
    if i == 0:
        break
print('---------------------End--------------------------')