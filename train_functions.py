#train functions for train.py
#Danny Olesh 22.11.2021

# Resources used:
# Study learning  notes and code from the course
# https://pytorch.org/
# Udacity deeplearning pytorch help
# Self study and experiminationion using ATOM in Anaconda3 environment



#################################################
#            imports                            #
#################################################
import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Subset
from collections import OrderedDict
import time
import json


#################################################
#            Get args and process               #
#################################################

def getOptions(args=sys.argv[2:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--save_dir", "--save_dir", help="directory for saving files.", default="")
    parser.add_argument("--arch", "--arch", help="Your destination output file.", default="vgg13")
    parser.add_argument("--learning_rate", "--learning_rate", type=float, help="learning rate .", default=0.01)
    parser.add_argument("--hidden_units", "--hidden_units", type=int, help="Hidden units number", default=512)
    parser.add_argument("--epochs", "--epochs", type=int, help="Epochs number", default=5)
    parser.add_argument("--gpu", dest='gpu',action='store_false', help="GPU mode True or False.", default=True)
    options = parser.parse_args(args)
    return options

#################################################
#            get data to transform              #
#################################################

def getDataTransforms(ns1, ns2, ns3, nm1, nm2, nm3, resize_num, centcrop_num):
    # arrays to normalization
    normalize_mean = np.array([nm1, nm2, nm3])
    normalize_std = np.array([ns1, ns2, ns3])

    data_transforms = {}

# transforms to train data set
    data_transforms['train'] = transforms.Compose([
                                        transforms.RandomChoice([
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(180),
                                        ]),
                                            transforms.RandomResizedCrop(centcrop_num),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                            normalize_mean,
                                            normalize_std)
                                            ])

# transforms to valid data set
    data_transforms['valid'] = transforms.Compose([
                                        transforms.Resize(resize_num),
                                        transforms.CenterCrop(centcrop_num),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        normalize_mean,
                                        normalize_std)
                                             ])
# transforms to test data set
    data_transforms['test'] = transforms.Compose([
                                      transforms.Resize(resize_num),
                                      transforms.CenterCrop(centcrop_num),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      normalize_mean,
                                      normalize_std)
                                            ])
    
    return  data_transforms

