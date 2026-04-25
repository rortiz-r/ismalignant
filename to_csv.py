import pandas as pd
import os

files_add = []

data_set_valid = ['Train', 'Test']
malignant = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'actinic keratosis']
benignant = ['nevus', 'dermatofibroma', 'pigmented benign keratosis', 'seborrheic keratosis', 'vascular lesion']


def get_binary_classification(classification_file):
    if classification_file in malignant:
        return  'malignant'
    return 'benignant'


for root, dirs, files in os.walk('./Skin cancer ISIC The International Skin Imaging Collaboration'):
    for file in files:
        path = os.path.join(root, file)
        classification = os.path.basename(os.path.dirname(path))
        data_set = os.path.basename(os.path.dirname(os.path.dirname(path)))
        
        file_to_add = {
            'path': path,
            'classification': get_binary_classification(classification),
            'data_set': data_set,
            'medical_name': classification
            }
        
        files_add.append(file_to_add)



dataset = pd.DataFrame(files_add)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset.to_csv('./data/dataset.csv')