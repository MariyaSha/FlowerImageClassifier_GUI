import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
from PIL import Image

device = torch.device('cpu')
#N top probabilities
top_k=5

def load_checkpoint(filepath):
    '''
    loads a model, trained with checkpoint.ipynb
    filepath: the path to a .pth file
    '''
    checkpoint = torch.load(filepath ,map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer_state = checkpoint['optimizer_state']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    '''takes in a PIL Image & transforms it to allow a forward pass'''
    image = image.resize((256,256))
    image = image.crop((0,0,224,224))
    np_image = np.array(image)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image = (np_image - mean) / std
    return pil_image

def load_parameters():
    '''Load checkpoint, and load class to index file into a dataframe'''
    #Loadings trained model for prediction
    model = load_checkpoint('checkpoint1.pth')

    #Loading a mapping of indexs numbers to flower names
    json_file = 'cat_to_name.json'
    with open(json_file, 'r') as f:
        idx_to_class = json.load(f)

    #Creating a Dataframe that contains Labels & Indices of Classes (df_combined)
    columns = {'Index' : pd.Series(model.class_to_idx),
                'Label': pd.Series(idx_to_class)}
    name_mapping = pd.DataFrame(columns)

    return model, name_mapping

def get_prediction(img):
    #get checkpoint and flower name mapping
    model, name_mapping = load_parameters()
    #load the image and process it
    image = Image.open(img)
    image = process_image(image)
    #get prediction (of top k probabilities)
    topk_ps = predict(image, model)
    #replace indices with names with name mappings
    topk_names = [name_mapping.values[i] for i in topk_ps]
    #print top k predictions to console
    print('\ntop ' + str(top_k) + ' predictions:')
    print_data = [print(i[1]) for i in topk_names]

    flower_name = topk_names[0][1] #return the top most probabilty
    return flower_name

#Prediction Function
def predict(image_path, model, top_k=top_k):
    '''returns the top k probabilities & classes for given user input image'''
    #convert image to tensor
    image_tensor = torch.from_numpy(image_path)
    image_tensor = np.transpose(image_tensor, (2,0,1))
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)
    #turn off gradients for forward pass
    with torch.no_grad():
        output = model.double().forward(image_tensor)
    #get top k probabilities
    ps = torch.exp(output)
    get_topk = ps.topk(top_k, sorted=True)

    topk_ps = get_topk[1][0] #return highest probability
    return topk_ps
