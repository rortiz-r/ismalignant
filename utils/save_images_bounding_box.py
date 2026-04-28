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

def load_image(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256,256))
    return img



def segmentate(img, model):
    img = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output).cpu().squeeze().numpy()
        mask = (output > 0.3).astype(np.uint8)

    return mask
    
def find_lesion_contours(blur_mask):
    (h,w) = blur_mask.shape[:2]
    
    mask_contours = np.zeros((h,w))

    contours, _ = cv.findContours(blur_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_max = max(contours, key=cv.contourArea)

    cv.drawContours(mask_contours, [contour_max], -1, 255, -1)
    
    return contour_max, mask_contours


def segmentate_lesion_crop_bounding_box(path, model):
    return segmentate(path, model)


def crop_image_bounding_box(mask_contours, image):
    x,y,w,h = cv.boundingRect(mask_contours.astype(np.uint8))
    
    crop_image = image[y:y+h, x:x+w]

    return crop_image





if __name__ == '__main__':
    

    # Guardar en nueva ubicación
    new_path = '../ISIC_BOUNDING_BOX_CROP'

    # Crear carpetas
    os.makedirs(new_path, exist_ok=True)
    os.makedirs(f"{new_path}/train", exist_ok=True)
    os.makedirs(f"{new_path}/test", exist_ok=True)
    os.makedirs(f"{new_path}/val", exist_ok=True)

    # read paths from csv.

    dataset = pd.read_csv('../data/01_dataset.csv')

    for index, row in  tqdm(dataset.iterrows(), total=len(dataset)):
        image = load_image(f".{row['path']}")
        mask = segmentate_lesion_crop_bounding_box(image, model)

        kernel = np.ones((15,15), np.uint8)
        morph_op = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
        morph_op = cv.morphologyEx(morph_op, cv.MORPH_ERODE, kernel, iterations=1)
        blur_mask = cv.blur(morph_op, (15,15)).astype(np.uint8)

        _, mask_contours = find_lesion_contours(blur_mask)

        image_crop = crop_image_bounding_box(mask_contours, image)

        # create 

        base_path = f"{new_path}/{row['data_set']}/{row['medical_name']}"

        os.makedirs(base_path, exist_ok=True)

        image_path = f"{base_path}/{row['base_id']}.jpg"

        cv.imwrite(image_path, cv.cvtColor(image_crop, cv.COLOR_RGB2BGR))


