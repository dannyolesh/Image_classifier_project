# train.py
# Danny Olesh 21th Nov 2021 V1.1
# 
# Resources used:
# Study learning  notes and code from the course
# https://pytorch.org/
# Udacity deeplearning pytorch help
# Self study and experiminationion using ATOM in Anaconda3 environment
# Edited code snippets for certain Network definitions https://github.com/silviomori/udacity-deeplearning-pytorch-challenge-lab


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
from train_functions import getOptions, getDataTransforms
from model_definitions import initializeModel 





print('Training system')



#################################################
#            Get args and process               #
#################################################
#get all the options required (after the fist 2, name and data_directory)


try:

    #assign data directory if argument is there
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        print('data directory needs to be entered usage: train.py data_directory')
        sys.exit(2)

    #check OS exists
    if os.path.exists(data_directory)==True:
        print('os Directory for data found: ', data_directory)
    else:
        print('data directory cannot be found please enter valid data directory')
        sys.exit(2)


    #remove first 2 args and load arguments
    options = getOptions(sys.argv[2:])
    
    # check save dir is valid
    if options.save_dir == "":
        options.save_dir = data_directory
        
    #assign GPU if true
    if options.gpu:
        print("GPU mode is on if available")
    else:
        print("GPU mode is off")

    if os.path.exists(options.save_dir)==True:
        print('Save directory :', options.save_dir)
    else:
        print('SAVE directory cannot be found please enter valid save directory')
        sys.exit(2)
    
    print('Architechutre chosen :', options.arch)
    if options.learning_rate < 1:
        print('Learning rate :', options.learning_rate)
    else:
        print('Learning rate seems high please restart with a smaller rate under 1')
        sys.exit(2)
        
        
    print('Hidden units :', options.hidden_units)

    print('Epoch number :', options.epochs)
   

except argparse.ArgumentError:
   print ('usage: train.py data_directory --save_dir <save directory> ')
   print ('--arch <architechture> --learning_rate <learning rate> ')
   print ('--hidden_units <hidden units> --epochs <number of epochs>')
   print ('--gpu <type --gpu to disable>')

   sys.exit(2)

# end getargs processing //////////////////////


#################################################
#           Assign some variables               #
#################################################

data_dir = data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



#######################################################################################
#           Define your transforms for the training and validation sets               #
#######################################################################################

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

# call data tranform function
data_transforms = getDataTransforms(ns1, ns2, ns3, nm1, nm2, nm3, resize_num, centcrop_num)

# Load the datasets with ImageFolder
image_datasets = {}
image_datasets['train_data'] = datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train'])
valid_dataset_to_split = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid'])


# obtain validation and training datasets that will be used to evaluate the network
valid_data_index_list = []
test_data_index_list = []
for index in range(0, len(valid_dataset_to_split), 2):
    valid_data_index_list.append(index)
    test_data_index_list.append(index+1)

image_datasets['valid_data'] = Subset(valid_dataset_to_split, valid_data_index_list)
image_datasets['test_data'] = Subset(valid_dataset_to_split, test_data_index_list)



# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {}
dataloaders['train_data'] = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True)
dataloaders['test_data'] = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
dataloaders['valid_data'] = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64)


#######################################################################################
#           Load the pictures into the system, process and display some to check      #
#######################################################################################

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
class_to_idx = image_datasets['train_data'].class_to_idx

cat_label_to_name = {}
for cat, label in class_to_idx.items():
    name = cat_to_name.get(cat)
    cat_label_to_name[label] = name
    
# Define imgview
#def imgview(img, title, ax):
    # un-normalize
#    for i in range(img.shape[0]):
 #       img[i] = img[i] * normalize_std[i] + normalize_mean[i]
    
    # convert from Tensor image
 #   ax.imshow(np.transpose(img, (1, 2, 0)))

#    ax.set_title(title)

# obtain one batch of training images

print('\nSelecting some pictures to train :')
print('')

dataiter = iter(dataloaders['train_data'])
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# show some test images
#fig = plt.figure(figsize=(15, 15))
fig_rows, fig_cols = 4, 5

# used on systems where we can display some pictures
for index in np.arange(fig_rows*fig_cols):


    label = labels[index].item()
    title = f'Label: {label}\n{cat_label_to_name[label].title()}'


#######################################################################################
#           Load the selected model and check the parameteres    #
#######################################################################################



#select Model to use

print('selecting model ', options.arch)



try:
    
       # get model attribute from list on Pytorch
    model_name = ''.join([i for i in options.arch if not i.isdigit()])
    
    # Initialize the model for this run
    output_size = 102
    out_features = options.hidden_units
    model_ft, input_size = initializeModel(options.arch, model_name, output_size, out_features, use_pretrained=True)

# Print the model we just instantiated
    print(model_ft)

    #if no model is able to get loaded
except AttributeError as error:
    print ('Model could not be loaded please try another model')
    sys.exit(2)
# loading model done



# USE GPU if available

device = torch.device("cuda" if torch.cuda.is_available() and options.gpu == True else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True. (from Pytorch)

params_to_update = model_ft.parameters()
print("Params to learn:")
feature_extract=True
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=options.learning_rate, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()



##################################################
#                       train the model          #     
##################################################



epochs = options.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in dataloaders['train_data']:
        steps += 1
        # Move input and label tensors to the default device
        #inputs, labels = inputs.to(device), labels.to(device)
        
        if device.type == 'cuda':
                inputs = inputs.float()
                labels = labels.long()
                inputs, labels = inputs.to(device), labels.to(device)
        else:
                inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model_ft.forward(inputs) # Forward propogation
        loss = criterion(logps, labels) # Calculates loss
        
        optimizer_ft.zero_grad() # zero's out the gradient
        loss.backward()  # Calculates gradient
        optimizer_ft.step() # Updates weights based on gradient

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model_ft.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['test_data']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model_ft.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(dataloaders['test_data']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['test_data']):.3f}")
            running_loss = 0
            model_ft.train()
            
            

            #######################################
            #      time to validate               #
            #######################################
# You can have data without information, but you cannot have information without data
#Daniel Keys Moran     



dataloader = dataloaders['test_data']


# Test the data
criterion = nn.NLLLoss()
test_loss = 0
accuracy = 0
top_class_graph = []
labels_graph = []


    # Set model to evaluation mode
model_ft.eval()    

with torch.no_grad():
        for images, labels in dataloader:
            
            labels_graph.extend( labels )

            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Get predictions for this test batch
            output = model_ft(images)

            # Calculate loss for this test batch
            batch_loss = criterion(output, labels)
            
            # Track validation loss
            test_loss += batch_loss.item()*len(images)

            # Calculate accuracy
            output = torch.exp(output)
            top_ps, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()


    # calculate average losses
test_loss = test_loss/len(dataloader.dataset)
accuracy = accuracy/len(dataloader.dataset)



 
     # print training/validation statistics 
log = f'Test Loss: {test_loss:.6f}\t\
    Test accuracy: {(accuracy*100):.2f}%'
print(log)


####################################
#  save save save                  #
####################################
model_name = options.arch
hidden_layers = options.hidden_units
        
def save_checkpoint(checkpoint_path, model_name, hidden_layers):
    model_ft.to('cpu')
    

        
    try:
    
        checkpoint = {'model_name': model_name,
                    'output_size': output_size,
                    'hidden_layers': hidden_layers,
                    'model_state_dict': model_ft.state_dict(),
                    'class_to_idx': class_to_idx,
                    'cat_label_to_name': cat_label_to_name}
        
        torch.save(checkpoint, checkpoint_path)
        
    except AttributeError as error: #no hidden layer in some
        checkpoint = {'model_name': model_name,
                    'output_size': output_size,
                    #'hidden_layers': options.hidden_units,
                    'model_state_dict': model_ft.state_dict(),
                    'class_to_idx': class_to_idx,
                    'cat_label_to_name': cat_label_to_name}
        
        torch.save(checkpoint, checkpoint_path)
        
        print('no hidden layer saved')
    

save_checkpoint(options.save_dir + '/checkpoint.pt', model_name, hidden_layers)

print(' ')
print('Checkpoint saved to:', options.save_dir + '/checkpoint.pt')