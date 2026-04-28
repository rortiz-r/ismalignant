import pandas as pd
import os
from sklearn.model_selection import train_test_split

files_add = []

data_set_valid = ['Train', 'Test']
malignant = ['melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'actinic keratosis']
benignant = ['nevus', 'dermatofibroma', 'pigmented benign keratosis', 'seborrheic keratosis', 'vascular lesion']


def get_binary_classification(classification_file):
    if classification_file in malignant:
        return  'malignant'
    return 'benignant'


for root, dirs, files in os.walk('./ISIC/'):
    for file in files:
        path = os.path.join(root, file)
        classification = os.path.basename(os.path.dirname(path))
        data_set = os.path.basename(os.path.dirname(os.path.dirname(path)))
        
        if file.endswith('.jpg'):
            file_to_add = {
            'path': path,
            'classification': get_binary_classification(classification),
            'medical_name': classification
            }
        
            files_add.append(file_to_add)



dataset = pd.DataFrame(files_add)

train, test = train_test_split(dataset, test_size=0.2, stratify=dataset['medical_name']) 

train['data_set'] = 'train'
test['data_set'] = 'test'

dataset_final = pd.concat([train, test])
print(dataset_final['data_set'].value_counts())
print(train['medical_name'].value_counts())
print(test['medical_name'].value_counts())
print(train['classification'].value_counts())
print(test['classification'].value_counts())
dataset_final = dataset_final.sample(frac=1).reset_index(drop=True)
dataset_final.to_csv('./data/dataset.csv')