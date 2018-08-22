# -*- coding: UTF-8 -*-
import torch
import torch.optim
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch.autograd import Variable

batch_size = 32
train_dataset = datasets.MNIST(
    root='./data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='./data/',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)
    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = x.view(in_size,-1)
        x = self.fc(x)
        return F.log_softmax(x)
model = CNN()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = Variable(data), Variable(target)
        print(target.size())
        optimizer.zero_grad()
        output = model(data)
        print(output.size())
        print(output)
        print(target)
        break
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(1,10):
        train(epoch)
        #test()
