import cv2
import numpy as np
import torch
from torchvision import transforms


def extract_features(img_path,model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(512),
        transforms.Resize(448),
        transforms.ToTensor()                              
    ])
    if torch.cuda.is_available():
        use_cuda=True
    else:
        use_cuda=False

    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    features = []
    img = cv2.imread(img_path)
    img = transform(img)
    img = img.reshape(1,3,448,448)
    img = img.to(device)
    with torch.no_grad():
        # extract fitur from image
        feature = model(img)
    features.append(feature.cpu().detach().numpy().reshape(-1))
    features = np.array(features)
    return features         
