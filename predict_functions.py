# predict functions for predict.py
# Danny Olesh 22.11.2021
# Resources used:
# Study learning  notes and code from the course
# https://pytorch.org/
# Udacity deeplearning pytorch help
# Self study and experiminationion using ATOM in Anaconda3 environment
# Edited code snippets for certain Network definitions https://github.com/silviomori/udacity-deeplearning-pytorch-challenge-lab


#################################################
#            imports                            #
#################################################
import sys
import os
from os.path import exists
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
from PIL import Image


#################################################
#            Get args and process               #
#################################################
#get all the options required (after the fist 3, predict.py /path/to/image /path/to/checkpoint)
def getOptions(args=sys.argv[3:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--top_k", "--top_k", type=int, help="Return top KK most likely classes", default=3)
    parser.add_argument("--catagory_names", "--catagory_names", help="Use a mapping of categories to real names(json file).", default="cat_to_name.json")
    parser.add_argument("--gpu", dest='gpu',action='store_false', help="GPU mode True or False.", default=True )
    options = parser.parse_args(args)
    
    if options.gpu: #assign GPU if true
        print("GPU mode is ON if available")
    else:
        print("GPU mode is OFF")

    if exists(options.catagory_names)==True: #check catagory name file exists
        print('Catagory names :', options.catagory_names)
    else:
        print('Catagory names file cannot be found please enter valid name')
        sys.exit(2)

    if options.top_k > 1 or options.top_k < 100: #check top k valid range
        print('The number of classes to compare :', options.top_k)
        
    else:
        print('top K needs to be a number between 1 and 100')
        sys.exit(2)
    
    return options


#################################################
#            Process an image                   #
#################################################

def process_image(image, normalize_mean, normalize_std):

    
    # Process a PIL image for use in a PyTorch model
    image = TF.resize(image, 256)
    
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)
    
    
    image = TF.to_tensor(image)
    image = TF.normalize(image, normalize_mean, normalize_std)
    
    return image

#################################################
#            Predictions                        #
#################################################

def predict(image_path, model_ft, topk,normalize_mean, normalize_std, device):
        
    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image, normalize_mean, normalize_std)
    
    with torch.no_grad():
        model_ft.eval()
        
        image = image.view(1,3,224,224)
        image = image.to(device)
        
        predictions = model_ft.forward(image)
        
        predictions = torch.exp(predictions)
        top_ps, top_class = predictions.topk(topk, dim=1)
    
    return top_ps, top_class
