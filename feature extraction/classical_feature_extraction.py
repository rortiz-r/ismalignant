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


# Load model
device = torch.device('mps')
model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load('./data/model_weights_skin_segmentation_ham10000.pth', map_location='mps'))

# Function loads and normalize image
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

    return area, perimeter, circularity

#Validate
def symmetry_score(mask_contours, area):
    mask_flip_y = cv.flip(mask_contours, 1)
    mask_flip_x = cv.flip(mask_contours, 0)

    symmetry_comparison_y = cv.absdiff(mask_contours, mask_flip_y)
    symmetry_comparison_x = cv.absdiff(mask_contours, mask_flip_x)

    total_y = np.sum(symmetry_comparison_y)/area
    total_x = np.sum(symmetry_comparison_x)/area

    return total_x, total_y


def color_variation(image, mask_contours):

    res = cv.bitwise_and(image, image, mask=mask_contours.astype(np.uint8))

    mask_image = cv.cvtColor(res, cv.COLOR_BGR2HSV)

    # Split color channels

    _,s,v = cv.split(mask_image)

    
    s = s[mask_contours>0]
    v = v[mask_contours>0]

    saturation_std = np.std(s)
    vue_std = np.std(v)


    return saturation_std, vue_std


def find_local_binary_pattern(image, mask_contours):
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8*radius
    lbp = local_binary_pattern(image_gray, n_points ,radius, method='uniform')
    lbp_filtered = lbp[mask_contours > 0]
    return np.histogram(lbp_filtered, bins=26)[0]


# def find_oriented_gradients(image):
#     image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#     fd, hog_img = hog(
#         image_gray, orientations=8
#     )


def extract_features(path):

    # Define feature dict

    features = {'diameter':None, 'area':None, 'perimeter':None, 'circularity':None, 'saturation_std':None, 'vue_std':None, 'total_x': None, 'total_y': None}


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

    area, perimeter, circularity = find_area_perimeter_circularity(contour_max)

    # Symmetry scores

    total_x, total_y = symmetry_score(mask_contours, area)
    
    # Color variation
    saturation_std, vue_std = color_variation(image, mask_contours)

    lbp = find_local_binary_pattern(image, mask_contours)

    features = {'diameter':diameter, 'area':area, 'perimeter':perimeter, 'circularity':circularity, 'saturation_std':saturation_std, 'vue_std':vue_std, 'total_x': total_x, 'total_y': total_y}
    
    for i in range(len(lbp)):
        features[f"p{i}"] = lbp[i]

    # Print original photo, mask and bounding rectangle

    # plt.figure(figsize=(10,10))
    # plt.subplot(1,2,1)
    # plt.title("Original image")
    # plt.imshow(image)

    # plt.subplot(1,2,2)
    # plt.title("Mask")
    # plt.imshow(mask, cmap='gray')
    

    # plt.subplot(2,2,1)
    # plt.title("Bounding rectangle")
    # plt.imshow(mask_bounding_rectangle, cmap='gray')
    # plt.show()

    return features





if __name__ == '__main__':

    dataset = pd.read_csv('./data/dataset.csv', index_col=0)
    print('Extracting features...')
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        features = extract_features(row['path'])
        for key, value in features.items():
            dataset.at[index, key] = value
        
    
    dataset.to_csv('./data/dataset_w_features.csv')

