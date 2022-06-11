import onnx
import crypten
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models
import os
import torch
import torch.nn as nn
import crypten.mpc as mpc
import crypten.communicator as comm
import BOT


#定义了计算调用的方法



def softmax(input, model):
    return model.forward(input)


def gemm(input, weight, bias):
    tmp = BOT.binary_OT(weight, bias, input)
    tmp.compt()
    return tmp.ans


def hardtanh(input, model):
    return model.forward(input)

def dropout(input, model):
    return model.forward(input)

def batchnormalization(input, model):
    return model.forward(input)

def relu(input, model):
    return model.forward(input)

none_para = {
    crypten.nn.module.Hardtanh : hardtanh,
    crypten.nn.module.Dropout : dropout,
    crypten.nn.module.ReLU : relu,
    crypten.nn.module.Softmax: softmax,
}

two_para = {
    crypten.nn.module.Gemm : gemm,
    crypten.nn.module.Linear : 'linear',
    crypten.nn.module.Conv : 'conv',
}

four_para = {
    crypten.nn.module.BatchNormalization : batchnormalization,

}
