#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path

from torch.autograd import Variable
import torch.onnx
import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.resnet50_unet_activation import UNetWithResnet50Encoder
from models.resnet50_unet_activation_DUA import UNetWithResnet50EncoderDUA
from models.resnet50_unet_activation_bilinear import UNetWithResnet50EncoderBi
from models.resnet50_unet_activation_drop import UNetWithResnet50Encoder_act_drop
from models.resnet50_unet_activation_no_bn import UNetWithResnet50Encoder_act_no_bn
from collections import OrderedDict


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state'].items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained_model', '-tm', type=str, help='trained model',
                        default='./checkpoints/glare_bestmodel_35_11_train_37_28.pth')
    parser.add_argument('--width', type=int, help='feature map width', default=3584)
    parser.add_argument('--height', type=int, help='feature map height', default=2560)
    parser.add_argument('--channel', type=int, help='feature map channel', default=3)
    args = parser.parse_args()

    model = UNetWithResnet50Encoder(n_classes=args.channel)
    
    # load it
    state_dict = torch.load(args.trained_model)
    model.load_state_dict(fix_model_state_dict(state_dict))
    x = Variable(torch.randn(1, args.channel, args.width, args.height))

    input_names_ = ["data"]
    output_names_ = ["pred"]
    torch.onnx.export(model, x, os.path.splitext(
        args.trained_model)[0] + '.onnx', verbose=True, input_names=input_names_, output_names=output_names_)
