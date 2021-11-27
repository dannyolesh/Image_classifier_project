# predict.py
# Danny Olesh 25th Nov 2021 V1.1
# Resources used:
# study learning  notes and code from the course
# https://pytorch.org/
# udacity deeplearning pytorch help
# self study and experiminationion using ATOM in Anaconda3 environment

#imports
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
from predict_functions import getOptions, predict, process_image
from model_definitions import initializeModel 
#from train_functions import initializeModel
from PIL import Image
from os.path import exists



print('Prediction system')


#################################################
#            Get args and process               #
#################################################

try:

    #assign /path/to/image and checkpoint if argument is there
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        checkpoint_path = sys.argv[2]
    else:
        print('image and  checkpoint path/names need to be entered, usage: predict.py /path/to/image /path/to/checkpoint')
        sys.exit(2)

    #check Image file exists
    image_path = '.' + image_path
    if exists(image_path) == True:
        print('image file found: ', image_path)
    else:
        print('image file cannot be found please enter valid path and name /path/to/image')
        sys.exit(2)

    #check checkpoint exists
    checkpoint_path = '.' + checkpoint_path
    if exists(checkpoint_path) == True:
        print('checkpoint file found: ', checkpoint_path)
    else:
        print('checkpoint file cannot be found please enter valid path and name  /path/to/checkpoint')
        sys.exit(2)

    #remove first 2 args and load arguments
    options = getOptions(sys.argv[3:])
    top_k = options.top_k

except argparse.ArgumentError:
   print ('usage: predict.py /path/to/image /path/to/checkpoint --top_k  <most likely classes list> ')
   print ('--catagory_names <jason file for catagory names> ')
   print ('--gpu <type --gpu to disable>')
   sys.exit(2)


#Variables

#np array numbers Normal
nm1 = 0.485
nm2 = 0.456
nm3 = 0.406

#np array numbers Standard
ns1 = 0.229
ns2 = 0.224
ns3 = 0.225

#size and crop
resize_num = 255
centcrop_num = 224


normalize_mean = np.array([nm1, nm2, nm3])
normalize_std = np.array([ns1, ns2, ns3])





def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    
    
    model_arch = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    
       # get model attribute from list on Pytorch
    model_name = ''.join([i for i in model_arch if not i.isdigit()])
    
    # Initialize the model for this run
    print('')
    print('model loaded:', model_arch)
    print('output size:', output_size)
    print('hidden layers:', hidden_layers)
    print('')
    
    
    model_ft, input_size = initializeModel(model_arch, model_name, output_size, hidden_layers, use_pretrained=True)   
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    model_ft.class_to_idx = checkpoint['class_to_idx']
    model_ft.cat_label_to_name = checkpoint['cat_label_to_name']
    
    
    return model_ft

model_ft = load_checkpoint(checkpoint_path)

# USE GPU if available
device = torch.device("cuda" if torch.cuda.is_available() and options.gpu == True else "cpu")


# Send the model to GPU or CPU
model_ft = model_ft.to(device)



# calculate image along with the top 5 classes
#probs, classes = predict(image_path, model_ft)

probs, classes = predict(image_path, model_ft, top_k, normalize_mean, normalize_std, device)


probs = probs.data.cpu()
probs = probs.numpy().squeeze()

classes = classes.data.cpu()
classes = classes.numpy().squeeze()
classes = [model_ft.cat_label_to_name[clazz].title() for clazz in classes]

try:
    image = Image.open(image_path)
except IOError:
    # filename not an image file
    print('Image file is not a valid image')
    sys.exit(2)
    
image = process_image(image, normalize_mean, normalize_std)

# Show final top K items in a nice way
print('The top ' + str(top_k) + ' classes and probability between those classes')
print('')
longest_name = max(classes, key=len) #calcualte length for proper display
max_lenght = len(longest_name) 
longest_number = len(str('{0:.2f}'.format(probs[0]))) #calcualte length for proper display


print('Number  Class name'.ljust(max_lenght, ' '), '\t\t', 'probability %') 
print('--------------------------------------------------------')
for kk in range (top_k): #cycle throught he top k requested
      #adjustments for proper display right and left alignment and spacing
      print('   ' + str(kk+1).rjust(2) + '   ' + classes[kk].ljust(max_lenght, ' '), '\t\t  ', ('{0:.2f}'.format(probs[kk]/probs.sum()*100)).rjust(longest_number))

exit()





