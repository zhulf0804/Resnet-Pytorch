import argparse
import os
import time

import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import torch.optim as optim
from torch.autograd import Variable

from resnet import resnet56

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', default=200, type=int, help="the training epoches number")
parser.add_argument('--saved_epoch', default=10, type=int, help='save checkpoints for this epoch number')
parser.add_argument('--print_epoch', default=2, type=int, help='print information for this epoch number')
parser.add_argument('--saved_dir', default='./checkpoint', type=str, help='the directory to save checkpoints')
parser.add_argument('--train_batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--test_batch_size', default=100, type=int, help='test batch size')
parser.add_argument('--resume', default=False, type=bool, help='whether to restore from checkpoint')
parser.add_argument('--restore_path', default='./', type=str, help='the path to restore parameters')
parser.add_argument('--data_dir', default='./data', type=str, help='the path to save data.')
parser.add_argument('--num_workers', default=4, type=int, help='the number of workers')
parser.add_argument('--init_lr', default=0.1, type=float, help='the initial learning rate to train the model')
parser.add_argument('--log_dir', default='./log', type=str, help='the dir to save training process')
parser.add_argument('--train_samples', default=50000, type=int, help='the number of training samples')
parser.add_argument('--test_samples', default=10000, type=int, help='the number of test samples')

args = parser.parse_args()


def train():

    if args.resume:
        restore_path = args.restore_path
        net.load_state_dict(t.load(restore_path))
        print("Parameters are restored from %s" %restore_path)
    else:
        print("Training for scratch...")

    if t.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.epoches + 1):
        #running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data  # Shape=(batch_size, channels, height, width), (batch_size, )
            inputs, lbales = Variable(inputs), Variable(labels)

            if t.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()


            outputs = net(inputs)

            loss = criterion(outputs, labels)
            writer.add_scalar('train_loss', loss.data.item(), i+epoch*(args.train_samples // args.train_batch_size + 1))
            optimizer.zero_grad()  # Important
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        if epoch % args.print_epoch == 0:
            print("Epoch: %d, Loss: %f" %(epoch, loss.data.item()))

        if epoch % args.saved_epoch == 0:
            if not os.path.exists(args.saved_dir):
                os.mkdir(args.saved_dir)
            t.save(net.state_dict(), os.path.join(args.saved_dir, 'resnet_%d.pth'%epoch))

            net.eval()
            coorect = 0
            total = 0
            losses = 0
            for data in testloader:
                images, labels = data

                images = Variable(images)
                labels = Variable(labels)

                if t.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                with t.no_grad():
                    outputs = net(images)
                    #optimizer.zero_grad()  # Important
                    loss = criterion(outputs, labels)

                losses += loss.data.item()
                _, predicted = t.max(outputs.data, 1)
                total += labels.size(0)
                coorect += (predicted == labels).sum()

            accuracy = coorect.double() * 1.0 / total
            print("Epoch: %d, Lr: %f, Total: %d, Correct: %d, Accuracy: %f, Test loss: %f" % (epoch, optimizer.param_groups[0]['lr'], total, coorect.double(), accuracy, losses / (args.test_samples / args.test_batch_size)))

            net.train()
            


# for dataset
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

transform_test = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = t.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = resnet56()

if t.cuda.is_available():
    net = net.cuda()


train()

#evaluate(net, testloader)