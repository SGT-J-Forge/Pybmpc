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
import time
import BOT
import common as cm

class ourCompute():
    def __new__(self, input, model):
        print('Initializing...')
        crypten.init()
        model.cuda()
        model.eval()
        dummy_input = torch.empty(input.shape).cuda()
        print('Encrypting model...')
        model_enc = crypten.nn.from_pytorch(model, dummy_input).encrypt()
        print('Encrypting input...')
        input = crypten.cryptensor(input)
        print('Create process Alice, process Bob...')
        output = input
        para = []
        sp = 0
        flag = 0

        for name, curr_module in model_enc._modules.items():
            if flag == 0:
                if type(curr_module) == crypten.nn.module.Parameter:
                    para.append(model_enc._modules[name].forward([]).get_plain_text())
                    continue
                elif type(curr_module) == crypten.nn.module.Constant:
                    sp = model_enc._modules[name].forward(output)
                    continue
                elif type(curr_module) == crypten.nn.module.Reshape:
                    output = model_enc._modules[name].forward(output, sp)
                    flag = 1
                    print('Initialization....Down')
                    print('-------------------MPC-------------------')
                    continue
            elif type(curr_module) in cm.none_para:
                print("Layer: ", curr_module)
                output = cm.none_para[type(curr_module)](output, model_enc._modules[name])
                continue
            elif type(curr_module) in cm.two_para:
                print("Layer: ", curr_module)
                output = model_enc._modules[name]([output, para.pop(0), para.pop(0)])
                continue
            elif type(curr_module) in cm.four_para:
                print("Layer: ", curr_module)
                output = cm.four_para[type(curr_module)]([output, para.pop(0), para.pop(0), para.pop(0), para.pop(0)],
                                                         model_enc._modules[name])
                continue
            else:
                print(f"{type(curr_module)} has not supported")
                return
        return output.get_plain_text()





