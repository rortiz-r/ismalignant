import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

files_add = []

data_set_valid = ['Train', 'Test']
malignant = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'actinic keratosis']
benignant = ['nevus', 'dermatofibroma', 'pigmented benign keratosis', 'seborrheic keratosis', 'vascular lesion']


def get_binary_classification(classification_file):
    if classification_file in malignant:
        return  'malignant'
    return 'benignant'


for root, dirs, files in os.walk('./ISIC_BOUNDING_BOX_CROP'):
    for file in files:
        path = os.path.join(root, file)
        classification = os.path.basename(os.path.dirname(path))
        data_set = os.path.basename(os.path.dirname(os.path.dirname(path)))
        
        if file.endswith('.jpg'):
            file_to_add = {
            'path': path,
            'classification': get_binary_classification(classification),
            'medical_name': classification,
            'base_id': Path(path).stem,
            'set_folder': data_set
            }
        
            files_add.append(file_to_add)



dataset = pd.DataFrame(files_add)

# train, test_temp = train_test_split(dataset, test_size=0.2, stratify=dataset['medical_name']) 

# test, val = train_test_split(test_temp, test_size=0.5, stratify=test_temp['medical_name']) 

# train['data_set'] = 'train'
# test['data_set'] = 'test'
# val['data_set'] = 'val'

# print(dataset_final['data_set'].value_counts())
# print(train['medical_name'].value_counts())
# print(test['medical_name'].value_counts())
# print(train['classification'].value_counts())
# print(test['classification'].value_counts())

#dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.drop_duplicates(subset=['base_id'], keep=False, inplace=False, ignore_index=False)
dataset.to_csv('./data/02_dataset_crop.csv')