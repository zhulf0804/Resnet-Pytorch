import argparse
import os
import time

import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from resnet import resnet56

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--restore_path', default='./checkpoint/resnet_180.pth', type=str, help='the path to restore parameters')
parser.add_argument('--data_dir', default='./data', type=str, help='the path to save data.')
parser.add_argument('--num_workers', default=4, type=int, help='the number of workers')

args = parser.parse_args()

def evaluate(net, testloader):

    net.eval()

    net_dict = net.load_state_dict(t.load(args.restore_path))

    coorect = 0
    total = 0
    for data in testloader:
        images, labels = data

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = net(images)

        _, predicted = t.max(outputs.data, 1)

        total += labels.size(0)

        coorect += (predicted == labels).sum()

    accuracy = coorect.double() * 1.0 / total
    print("Total: %d, Correct: %d, Accuracy: %f" % (total, coorect.double(), accuracy))


# for dataset
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = resnet56()
net = net.cuda()

evaluate(net, testloader)