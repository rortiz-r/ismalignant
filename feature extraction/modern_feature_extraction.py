import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models 
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
model = models.resnet50(weights='DEFAULT').to(device)
model.fc = nn.Identity()

class CustomDataset(Dataset):
    def __init__(self, img):
        self.imgs = img

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = cv.imread(f'{self.imgs[index]}') # Improve
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (256,256))
        img = img.astype('float32') / 255.0

        img = torch.tensor(img).permute(2,0,1)
        
        return img


def get_features(image):
    with torch.no_grad():
        output = model(image).cpu().numpy()

    return output


if __name__ == '__main__':

    dataset = pd.read_csv('../data/02_dataset_crop.csv')
    dataset = dataset.sort_values('image_id').reset_index(drop=True)

    img_paths = dataset['path']

    img_paths = np.array(img_paths)
    img_base_ids = np.array(dataset['image_id'])

    images_dataset = CustomDataset(img_paths)
    img_loader = DataLoader(images_dataset, batch_size=32, shuffle=False, num_workers=4)

    embeddings = []

    model.eval()
    for idx, data in enumerate(tqdm(img_loader)):
        embedding = get_features(data.to(device))
        embeddings.extend(embedding)



    # convert to np array

    embeddings_np = np.array(embeddings)

    print(embeddings_np.shape)

    np.savez('../data/embeddings.npz', embeddings = embeddings_np, base_ids=img_base_ids)

        



    

