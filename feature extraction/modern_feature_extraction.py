import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
from glob import glob
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from transformers import get_scheduler
import evaluate
from tqdm import tqdm



device = torch.device('mps')
model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load('../data/model_weights_skin_segmentation_ham10000.pth', map_location='mps'))

def segmentate(image_path, model):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256,256))
    img = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output).cpu().squeeze().numpy()
        mask = (output > 0.3).astype(np.uint8)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')

    return mask
    

def segmentate_lesion(path, model):
    mask = segmentate(path, model)
    mask_bounding_rectangle = np.zeros((h,w))

    
    x,y,w,h = cv.boundingRect(mask_contours.astype(np.uint8))
    cv.rectangle(mask_bounding_rectangle, (x,y), (x+w, y+h), 255)
    plt.imshow(mask_bounding_rectangle, cmap='gray')
    diameter = max(w,h)
    print(f"El diametro es: {diameter}")


class CustomDataset(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = cv.imread(self.imgs[index])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256,256))
        img = img.astype('float32') / 255.0

        img = torch.tensor(img).permute(2,0,1)
        
        img_mask = cv.imread(self.masks[index], cv.IMREAD_GRAYSCALE)
        img_mask = cv.resize(img_mask, (256,256))
        img_mask = img_mask.astype('float32') / 255.0
        img_mask = np.expand_dims(img_mask, axis=0)

        img_mask = torch.tensor(img_mask)


        return img,img_mask



if __name__ == '__main__':
    path = './ISIC/Train/dermatofibroma/ISIC_0027044.jpg'
    segmentate_lesion(path, model)
