import argparse
import os
import cv2
import tifffile
import time

import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from resnet import resnet56

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--restore_path', default='./checkpoint/resnet_180.pth', type=str, help='the path to restore parameters')
parser.add_argument('--data_dir', default='./data', type=str, help='the path to save data.')
parser.add_argument('--num_workers', default=4, type=int, help='the number of workers')

args = parser.parse_args()

def evaluate(net, testloader):

    net.eval()

    if t.cuda.is_available():
        net_dict = net.load_state_dict(t.load(args.restore_path))
    else:
        net_dict = net.load_state_dict(t.load(args.restore_path, map_location='cpu'))

    correct = 0
    total = 0
    for i, data in enumerate(testloader):
        images, labels = data
        if i < 10:
            img = images[0].permute(1, 2, 0).detach().cpu().numpy()
            label = labels[0].detach()
            tifffile.imwrite('data/{}_{}.tiff'.format(i, label), img)
        images = Variable(images)
        labels = Variable(labels)

        if t.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)

        _, predicted = t.max(outputs.data, 1)

        if i < 10:
            print(i, outputs[0].detach().cpu().numpy(), predicted[0].detach().cpu().numpy(), labels[0].detach().cpu().numpy())

        total += labels.size(0)

        correct += (predicted == labels).sum()

    accuracy = correct.double() * 1.0 / total
    print("Total: %d, Correct: %d, Accuracy: %f" % (total, correct.double(), accuracy))


# for dataset
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = resnet56()
if t.cuda.is_available():
    net = net.cuda()

evaluate(net, testloader)