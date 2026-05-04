import os
from glob import glob
import pandas as pd
from ..config import HOME_DIR, BASE_PATH




def dx_column(dataset):
    for index, row in dataset.iterrows():
        if row['NV'] == 1.0:
            dataset.at[index, 'dx'] = 'NV'
        else:
            continue
    return dataset

def get_metadata(path, crop=False):
    metadata = pd.read_csv(path)

    if crop :
        return metadata[['image', 'dx', 'path']]
    
    mask = (metadata['MEL'] == 1.0) | (metadata['NV'] == 1.0)
    metadata = metadata[mask]

    metadata['dx'] = 'MEL'

    metadata_dx = dx_column(metadata) 
    

    return metadata_dx[['image', 'dx']]


if __name__ == '__main__':

    dataset_path = '/Downloads/ISIC_2019_BOUNDING_BOX_CROP'
    images_path = f"{dataset_path}"
    abs_data_path = f"{HOME_DIR}/{images_path}"

    ground_truth = f'{BASE_PATH}/data/02_dataset_w_features.csv'

    metadata = get_metadata(ground_truth, crop=True)

    
    # # Crear paths
    for index, row in metadata.iterrows():
        full_path = f'{images_path}/{row['dx']}/{row['image']}.jpg'
        metadata.at[index, 'path'] = full_path


    save_path = f'{BASE_PATH}/data/03_dataset_crop.csv'
    
    metadata.to_csv(save_path, index=False)