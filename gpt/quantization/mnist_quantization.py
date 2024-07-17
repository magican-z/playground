#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import pickle
import numpy as np
import torch
from torch import nn
import torch.ao.quantization
import torch.nn.functional as F


class Mnist(object):
    train_num = 60000
    test_num = 10000
    img_dim = (1, 28, 28)
    img_size = 784

    def load_mnist(self, save_file, normalize=True, flatten=True, one_hot_label=False):
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if one_hot_label:
            dataset['train_label'] = self._change_one_hot_label(dataset['train_label'])
            dataset['test_label'] = self._change_one_hot_label(dataset['test_label'])

        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

    def _change_one_hot_label(self, X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
        return T


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Linear(784, 98)
        self.n2 = nn.Linear(98, 10)

    def forward(self, x):
        x = self.n1(x)
        x = self.n2(F.relu(x))
        return x


class MLP_Q(nn.Module):
    def __init__(self, mlp, qw_type=torch.int8, qw_width=8):
        super().__init__()
        assert(2 ** (qw_width - 1) - 1 <= torch.iinfo(qw_type).max)
        self.mlp = mlp
        self.qw_type = qw_type
        self.qa_type = torch.int16
        self.qw_max = 2 ** (qw_width - 1) - 1
        self.quantify_weights()

    def quantify(self, x, x2=None):
        max_v = torch.max(torch.abs(x)).item()
        if x2 is not None:
            max_v2 = torch.max(torch.abs(x2)).item()
            max_v = max(max_v, max_v2)
        scale = max_v / self.qw_max
        x_q = torch.round(x / scale).to(self.qw_type)
        if x2 is not None:
            x_q = (x_q, torch.round(x2 / scale).to(self.qw_type))
        return x_q, scale
    
    def unquantify(self, x, scale):
        return x.to(torch.float32) * scale
    
    def quantify_weights(self):
        (self.w1, self.b1), self.scale_1 = self.quantify(self.mlp.n1.weight.data.T, self.mlp.n1.bias.data)
        (self.w2, self.b2), self.scale_2 = self.quantify(self.mlp.n2.weight.data.T, self.mlp.n2.bias.data)
        

    def dot_product(self, q_x1, q_x2, scale_1, scale_2, target_scale):
        y0 = q_x1.to(torch.int32) @ q_x2.to(torch.int32)
        scale = scale_1 * scale_2 / target_scale
        y = y0 * scale
        return torch.round(y).to(self.qa_type)
    
    def forward(self, x):
        x_q, x_scale = self.quantify(x)
        o1 = self.dot_product(x_q, self.w1, x_scale, self.scale_1, self.scale_1) + self.b1.to(self.qa_type)
        o2 = self.dot_product(F.relu(o1), self.w2, self.scale_1, self.scale_2, self.scale_2) + self.b2.to(self.qa_type)
        return self.unquantify(o2, self.scale_2)


class MLP_PQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.n1 = nn.Linear(784, 98)
        self.relu = torch.nn.ReLU()
        self.n2 = nn.Linear(98, 10)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.n2(x)
        x = self.dequant(x)
        return x


def quantize_by_torch(model, samples):
    q_model = MLP_PQ()
    q_model.n1 = model.n1
    q_model.n2 = model.n2
    q_model.eval()
    q_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    model_fp32_fused = torch.ao.quantization.fuse_modules(q_model, [['n1', 'relu']])
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

    # 校准模型
    model_fp32_prepared(samples)

    # 转换模型为量化模型
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    return model_int8



def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)



if __name__ == "__main__":
    torch.set_grad_enabled(False)
    data_loader = Mnist()
    (train_img, train_label), (test_img, test_label) = data_loader.load_mnist(sys.argv[1])
    train_img = torch.tensor(train_img, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.uint8)
    train_img = torch.tensor(train_img, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.uint8)
    test_img = torch.tensor(test_img, dtype=torch.float32)
    test_label = torch.tensor(test_label,  dtype=torch.uint8)

    # 初始模型
    checkpoint = torch.load(sys.argv[2])
    raw_model = MLP()
    raw_model.load_state_dict(checkpoint['model'])
    raw_pred_logits = raw_model(test_img)
    raw_pred_y = F.softmax(raw_pred_logits, dim=-1)
    raw_acc = accuracy(raw_pred_y, test_label)

    # 自己实现的量化模型
    q_model = MLP_Q(raw_model, qw_width=8)
    q_pred_logits = q_model(test_img)
    q_pred_y = F.softmax(q_pred_logits.to(torch.float32), dim=-1)
    q_acc = accuracy(q_pred_y, test_label)

    # torch.quantization实现的量化
    tq_model = quantize_by_torch(raw_model, train_img[:200])
    tq_pred_logits = tq_model(test_img)
    tq_pred_y = F.softmax(tq_pred_logits, dim=-1)
    tq_acc = accuracy(tq_pred_y, test_label)

    print(f'raw_acc: {raw_acc}, my_quantize_acc: {q_acc}, torch_quantize_acc: {tq_acc}')

    
