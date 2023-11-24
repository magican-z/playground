#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import torchvision
from torchvision import transforms
import torch
from torch import nn
import json


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=4),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=4))


def get_mean_l1(grad):
    # 计算梯度的L1范数，然后求均值
    return float(torch.norm(grad, p=1) / torch.numel(grad))


def output_grad(net, i):
    # 这个函数要研究具体的网络结构，根据网络结构去取权重的梯度
    layers_grad = []
    layers_grad.append(get_mean_l1(net.conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer1[0].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer1[0].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer1[1].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer1[1].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer2[0].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer2[0].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer2[1].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer2[1].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer3[0].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer3[0].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer3[1].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer3[1].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer4[0].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer4[0].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.layer4[1].conv1.weight.grad))
    layers_grad.append(get_mean_l1(net.layer4[1].conv2.weight.grad))
    layers_grad.append(get_mean_l1(net.fc.weight.grad))
    print('Setp:{}\t{}'.format(i, json.dumps(layers_grad)))


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            output_grad(net, i)
            optimizer.step()


def get_devices(i=0):
    return torch.device('cuda:{}'.format(i))


if __name__ == "__main__":
    net = torchvision.models.resnet18()
    # 适配fashion_minst的输入输出：输出10类，输入的分辨率是64
    net.fc = nn.Linear(512,10)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    lr, num_epochs, batch_size = 0.05, 5, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train(net, train_iter, test_iter, num_epochs, lr, get_devices())
