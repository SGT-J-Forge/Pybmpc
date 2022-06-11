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


class binary_OT():
    """
    Binary Obviously Transfer
    """

    def __init__(self, weight, bias, x):
        self.ans = 0
        self.weight = weight
        self.bias = bias.view(1, bias.shape[0])
        self.x = x.get_plain_text()


    def compt(self):
        self.kb = bob_init(self.weight)
        self.ka = alice_init(self.kb, self.weight)
        self.data_send, self.x_bob = bob_compt(self.x, self.kb)
        self.x_alice = alice_compt(self.weight, self.data_send)
        self.ans = crypten.cryptensor(self.x_alice + self.x_bob + self.bias)



def bob_init(weight):
    kb_0 = torch.rand_like(weight)
    kb_1 = torch.rand_like(weight)
    kb = []
    kb.append(kb_0)
    kb.append(kb_1)
    return kb

def alice_init(kb,weight):
    ka = torch.where(
        weight == 1,
        kb[1],
        kb[0]
    )
    return ka

def bob_compt(input,kb,batchsize=1):
    output_dim = kb[0].shape[0]
    input_dim = kb[0].shape[1]
    r = torch.randn(output_dim, input_dim)
    data_send = [
        r + input,
        r - input
    ]
    mult = torch.ones(input_dim, batchsize)
    x_bob = torch.mm(r, -mult).t()
    return data_send, x_bob

def alice_compt(weight, data_rev,batchsize=1):
    mult = torch.ones(weight.shape[1], batchsize)
    x = torch.where(
        weight == 1,
        data_rev[0],
        data_rev[1]
    )
    x_alice = torch.mm(x, mult).t()
    return x_alice

