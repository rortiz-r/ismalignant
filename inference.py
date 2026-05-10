
from ismalignant.feature_extraction.classical_feature_extraction import extract_features, load_image
from ismalignant.feature_extraction.embeddings_extraction import get_features, CustomDataset
from ismalignant.recommendation.similar_lesion import get_recommendations
import sys
import numpy as np
import pandas as pd
from .config import load_scaler_pca_svm, device, BASE_PATH, HOME_DIR, model_resnet
import matplotlib.pyplot as plt
from pathlib import Path




def get_classic_modern_features(path):

    classic_features =  extract_features(path, True)

    image_dataset = CustomDataset([path])

    embeddings = get_features(image_dataset[0].unsqueeze(0).to(device), model_resnet)


    return classic_features, embeddings


def process_data_for_model(classic_features, embeddings):

    image_classic_features = scaler_classic.transform(classic_features)
    embeddings_features = scaler_cnn.transform(embeddings)

    image_features = np.hstack([image_classic_features, embeddings_features])

    return image_features


def format_classification(prediction, t):
    if prediction >= t:
        return 'Melanoma'
    return 'Nevo'

def get_percentiles(data, feature):
    return data[feature][0], data[feature][0], data[feature][0], data[feature][0]


def get_lecture_for_circularity(value, p25, p75, p90):

    if value < p25:
        return 'highly irregular borders'
    elif value < p75:
        return 'irregular borders'

    elif value < p90:
        return 'slightly regular borders'
    
    return 'highly regular borders'

def get_lecture_for_diameter(value, p25, p75, p90):

    if value < p25:
        return 'small lesion'
    elif value < p75:
        return 'medium lesion'

    elif value < p90:
        return 'large lesion'
    
    return 'big lesion'

def get_lecture_for_asymmetry(value, p25, p75, p90):

    if value < p25:
        return 'slightly asymmetrical lesion'
    elif value < p75:
        return 'moderately asymmetrical lesion'
    elif value < p90:
        return 'highly asymmetrical lesion'
    return 'very asymmetrical lesion'

def get_percentile_lecture(value, p25, p75, p90):
    if value < p25:
        return 'lower than the 25%'
    elif value < p75:
        return 'lower than the 75%'
    elif value < p90:
        return 'lower than the 90%'
    
    return 'higher than the 90%'

def get_features_percentile_analysis(features_names, features_values, data, type):
    
    for i in range(len(features_names)):
        p25,_, p75, p90 = get_percentiles(data, features_names[i])
        if features_names[i] == 'diameter':
            lecture = get_lecture_for_diameter(features_values[0][i], p25, p75, p90)
        elif features_names[i] == 'circularity':
            lecture = get_lecture_for_circularity(features_values[0][i], p25, p75, p90)
        elif features_names[i] == 'total_x':
            lecture = get_lecture_for_asymmetry(features_values[0][i], p25, p75, p90)
        else:
            lecture = get_lecture_for_asymmetry(features_values[0][i], p25, p75, p90)

        percentile_lecture = get_percentile_lecture(features_values[0][i], p25, p75, p90)
        print(f"{features_names[i]} {features_values[0][i]:.3f} {lecture}\n")


def show_recommendations(recommendations):

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    fig.suptitle('Similar Lesions', fontsize=22)
    for i, (_, row) in enumerate(recommendations.iterrows()):
        img_path = f'{HOME_DIR}{row['path']}'
        img = load_image(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"{row['image']}\n{row['dx']}", fontsize=9)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # 1.Recieve path on terminal
    path = sys.argv[1]
    
    classic_features, embeddings = get_classic_modern_features(path)
    feature_names = ['diameter', 'circularity', 'total_x', 'total_y', 'saturation_mean', 'saturation_std', 'val_mean', 'val_std'] + [f'h{i}' for i in range(4)]

    classic_features_val = np.array([classic_features[k] for k in feature_names if k in classic_features]).reshape(1,len(feature_names))


    scaler_classic, scaler_cnn, label_encoder, svm = load_scaler_pca_svm()
    
    features = process_data_for_model(classic_features_val, embeddings)
    t = 0.5
    prob = svm.predict_proba(features)
    prob_index = np.argmax(prob[0] >= t)
    confidence = (prob[0][prob_index]) * 100
    prediction = format_classification(prob[0][prob_index], t=t)

    if confidence <= 70:
        print("Mixed clinical features")


    # SIMILAR LESIONS

    similar_lesions = get_recommendations(features)

    
    ## Statistics by class

    melanoma_p = pd.read_csv(f'{BASE_PATH}/data/melanoma_percentiles.csv')
    nevo_p = pd.read_csv(f'{BASE_PATH}/data/nevo_percentiles.csv')

    ### Formatting output
    print(f"The lesion is likely a {prediction} with a confidence of {confidence:.2f}%")
    print("Comparison between melanomas")

    interpretation = ['diameter', 'circularity', 'total_x', 'total_y']
    interpretation_features_val = np.array([classic_features[k] for k in interpretation if k in classic_features]).reshape(1,len(interpretation))

    get_features_percentile_analysis(interpretation, interpretation_features_val, melanoma_p, 'Melanoma')
    print("Comparison between nevo")
    get_features_percentile_analysis(interpretation, interpretation_features_val, nevo_p, 'Nevus')
    show_recommendations(similar_lesions)
    


