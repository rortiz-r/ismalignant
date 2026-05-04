import pandas as pd
import numpy as np
import torch
import cv2 as cv
import math
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from ..config import device, model, BASE_PATH, HOME_DIR
import os
from pathlib import Path


# Function loads and normalize image
def load_image(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256,256))
    return img


def is_well_segmented(contour):
    area = cv.contourArea(contour)
    if len(contour) < 5 or area < 0.01:
        return False

    return True


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

    if not contours:
        return None, None

    contour_max = max(contours, key=cv.contourArea)

    cv.drawContours(mask_contours, [contour_max], -1, 255, -1)
    
    return contour_max, mask_contours


def find_diameter(blur_mask):
    (h,w) = blur_mask.shape[:2]
    x,y,w,h = cv.boundingRect(blur_mask.astype(np.uint8))
    diameter = max(w,h)
    return diameter


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

    features = {'diameter': None, 'compactness': None, 'circularity': None, 'saturation_std': None, 'val_std':None, 'total_x': None, 'total_y': None, 'saturation_mean': None, 'val_mean': None, 'entropy_h': None, 'entropy_s': None, 'entropy_v':None}
    

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

    if contour_max is None or not is_well_segmented(contour_max):
        return features
    
    # Find the perimeter of the lesion

    diameter = find_diameter(mask_contours)

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

    dataset_path = f"{BASE_PATH}/data/01_dataset.csv"
    
    dataset = pd.read_csv(dataset_path, index_col=0)

    dataset = dataset.reset_index()

    dataset = dataset.sort_values('image')


    print('Extracting features...')
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):

        path = f"{HOME_DIR}/{row['path']}"
        features = extract_features(path)

        for key, value in features.items():
            dataset.at[index, key] = value
        
    
    print(dataset['dx'].value_counts())
    # Filter dataset

    # Filter rows where total_y and total_x are 0
    mask = (dataset['total_x'] == 0.0) & (dataset['total_y'] == 0.0)
    dataset = dataset[~mask].dropna()
    dataset = dataset.drop_duplicates(subset=['image'], keep=False, inplace=False, ignore_index=False)

    print(dataset['dx'].value_counts())

    save_path = f'{BASE_PATH}/data/02_dataset_w_features.csv'

    dataset.to_csv(save_path)


