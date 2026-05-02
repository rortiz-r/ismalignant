import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
import os

# Load model
device = torch.device('mps')
model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load('../data/model_weights_skin_segmentation_ham10000.pth', map_location='mps'))
model.eval()

# Function loads and normalize image
def load_image(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256,256))
    return img


def segmentate(img, model):
    img = img.astype('float32') / 255.0
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device)

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


def find_diameter(blur_mask):
    (h,w) = blur_mask.shape[:2]
    mask_bounding_rectangle = np.zeros((h,w))
    x,y,w,h = cv.boundingRect(blur_mask.astype(np.uint8))
    cv.rectangle(mask_bounding_rectangle, (x,y), (x+w, y+h), 255)
    diameter = max(w,h)
    return mask_bounding_rectangle, diameter


def find_area_perimeter_circularity(contour_max):
    perimeter = cv.arcLength(contour_max, True)
    area = cv.contourArea(contour_max)

    # Raise exception.
    if perimeter > 0:
        circularity = ((4*math.pi*area)/(perimeter*perimeter))
        compactness = (perimeter*perimeter)/area

    return circularity, compactness

#Validate
def symmetry_score(mask_contours, contour_max):

    if len(contour_max) < 5:
        return None, None

    ((h_mask, w_mask)) = mask_contours.shape[:2]

    # Find the center of the lesion

    moments = cv.moments(mask_contours)

    # Centroid

    cX = int(moments['m10']/moments['m00'])
    cY = int(moments['m01']/moments['m00'])

    # Angle

    angle = cv.fitEllipse(contour_max)[2]


    rotation_matrix = cv.getRotationMatrix2D((cX, cY), 90-angle, 1.0)


    rotated = cv.warpAffine(mask_contours, rotation_matrix, (h_mask, w_mask))

    # mirrored

    mirrored_y = cv.flip(rotated, 1)
    mirrored_x = cv.flip(rotated, 0)


    # Intersection

    inter_y = np.sum(cv.bitwise_and(rotated, mirrored_y) > 0)

    # Union

    union_y = np.sum(cv.bitwise_or(rotated, mirrored_y) > 0)



    asymmetry_index_y = inter_y / union_y


    ####


    inter_x = np.sum(cv.bitwise_and(rotated, mirrored_x) > 0)

    # Union

    union_x = np.sum(cv.bitwise_or(rotated, mirrored_x) > 0)



    asymmetry_index_x = inter_x / union_x




    return asymmetry_index_x, asymmetry_index_y


def color_variation(image, mask_contours):

    res = cv.bitwise_and(image, image, mask=mask_contours.astype(np.uint8))

    mask_image = cv.cvtColor(res, cv.COLOR_RGB2HSV)

    # Split color channels

    _,s,v = cv.split(mask_image)

    
    s = s[mask_contours>0]
    v = v[mask_contours>0]

    saturation_std = np.std(s)
    val_std = np.std(v)

    saturation_mean = np.mean(s)
    val_mean = np.mean(v)

    # Test color hist

    hist_h = cv.calcHist([mask_image], [0], None, [256], [0,256])
    hist_h = hist_h.ravel() / hist_h.sum()

    probs_h = hist_h[hist_h > 0]

    hist_s = cv.calcHist([mask_image], [1], None, [256], [0,256])
    hist_s = hist_s.ravel() / hist_s.sum()

    probs_s = hist_s[hist_s > 0]

    hist_v = cv.calcHist([mask_image], [2], None, [256], [0,256])
    hist_v = hist_v.ravel() / hist_v.sum()

    probs_v = hist_v[hist_v > 0]

    entropy_h = - np.sum(probs_h * np.log2(probs_h))
    entropy_s = - np.sum(probs_s * np.log2(probs_s))
    entropy_v = - np.sum(probs_v * np.log2(probs_v))


    return saturation_std, val_std, saturation_mean, val_mean, entropy_h, entropy_s, entropy_v


def find_local_binary_pattern(image, mask_contours):
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8*radius
    lbp = local_binary_pattern(image_gray, n_points ,radius, method='uniform')
    lbp_filtered = lbp[mask_contours > 0]
    return np.histogram(lbp_filtered, bins=26)[0]



def extract_features(path):

    image = load_image(path)
    mask = segmentate(image, model)



    if np.sum(mask) == 0:
        return features 
    
    # Apply morphological operations and apply blur to mask
    kernel = np.ones((15,15), np.uint8)
    morph_op = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel, iterations=1)
    morph_op = cv.morphologyEx(morph_op, cv.MORPH_ERODE, kernel, iterations=1)
    blur_mask = cv.blur(morph_op, (15,15)).astype(np.uint8)

    #Find the contour with the biggest area.
    # Mask contours contains the mask with the relevant area.
    
    contour_max, mask_contours = find_lesion_contours(blur_mask)

    
    # Find the perimeter of the lesion

    mask_bounding_rectangle, diameter = find_diameter(mask_contours)

    # Calculate area and perimeter and circularity

    circularity, compactness = find_area_perimeter_circularity(contour_max)

    # Symmetry scores

    total_x, total_y = symmetry_score(mask_contours, contour_max)
    
    # Color variation
    saturation_std, val_std, saturation_mean, val_mean, entropy_h, entropy_s, entropy_v = color_variation(image, mask_contours)

    lbp = find_local_binary_pattern(image, mask_contours)

    features = {'diameter':diameter, 'compactness': compactness, 'circularity':circularity, 'saturation_std':saturation_std, 'val_std':val_std, 'total_x': total_x, 'total_y': total_y, 'saturation_mean': saturation_mean, 'val_mean': val_mean, 'entropy_h': entropy_h, 'entropy_s': entropy_s, 'entropy_v':entropy_v}
    
    for i in range(len(lbp)):
        features[f"p{i}"] = lbp[i]


    return features





if __name__ == '__main__':

    dataset = pd.read_csv('../data/01_dataset.csv', index_col=0)
    dataset = dataset.sort_values('image_id').reset_index(drop=True)

    print('Extracting features...')
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        path = f"{row['path']}"
        features = extract_features(path)

        for key, value in features.items():
            dataset.at[index, key] = value
        
    
    dataset.to_csv('../data/03_dataset_w_features.csv')


