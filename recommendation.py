import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Any results you write to the current directory are saved as output.
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import time
from PIL import Image

import cv2
import torch
from torch import optim, nn
from torchvision import models, transforms

import torchvision
from torchvision import transforms

if torch.cuda.is_available():
    use_cuda=True
else:
    use_cuda=False

device = torch.device("cuda:0" if use_cuda else "cpu")


# remake the model 
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Initialize the model
from FeatureExtractor import FeatureExtractor
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

st.title("Fashion Recommender System")

# uploaded images
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb')as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

def extract_features(img_path,model):
    features = []
    img = cv2.imread(img_path)
    img = transform(img)
    img = img.reshape(1,3,448,448)
    img = img.to(device)
    with torch.no_grad():
        # extract fitur from image
        feature = new_model(img)
    features.append(feature.cpu().detach().numpy().reshape(-1))
    features = np.array(features)
    return features         

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distances,indices = neighbors.kneighbors([features])
    
    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
#jika tidak kosong
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #feature extract
        features = extract_features(os.path.join('uploads',uploaded_file.name),new_model)
        st.text(features.shape)
        #recomendation
        indices = recommend(features.squeeze(),feature_list.squeeze())
        #show
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")

