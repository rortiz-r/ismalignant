
from feature_extraction.classical_feature_extraction import extract_features
from feature_extraction.modern_feature_extraction import get_features, CustomDataset
import sys
import torch
import joblib
import numpy as np
import pandas as pd


device = torch.device('mps')

def load_scaler_pca_svm():
    scaler_classic = joblib.load('./data/scaler_classic.pkl')
    scaler_cnn = joblib.load('./data/scaler_cnn.pkl')
    pca = joblib.load('./data/pca.pkl')
    svm = joblib.load('./data/model_svm_wo_lbp_and_colors_4.pkl')
    label_encoder = joblib.load('./data/label_encoder.pkl')

    return scaler_classic, scaler_cnn, pca, label_encoder,svm

def get_classic_modern_features(path):

    classic_features =  extract_features(path)

    image_dataset = CustomDataset([path])

    embeddings = get_features(image_dataset[0].unsqueeze(0).to(device))

    return classic_features, embeddings

def process_data_for_model(classic_features, embeddings):

    image_classic_features = scaler_classic.transform(classic_features)
    embeddings_features = scaler_cnn.transform(embeddings)

    embeddings_features = pca.transform(embeddings_features)

    image_features = np.hstack([image_classic_features, embeddings_features])

    return image_features


def format_classification(prediction):
    if prediction > 0.5:
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
        print(f"{features_names[i]} ({features_values[0][i]:.3f}): {percentile_lecture} of {type} lesions — {lecture}\n")


if __name__ == '__main__':

    # 1.Recieve path on terminal
    path = sys.argv[1]
    
    classic_features, embeddings = get_classic_modern_features(path)
    feature_names = ['diameter', 'circularity', 'total_x', 'total_y']


    classic_features_val = [classic_features[k] for k in feature_names if k in classic_features]
    classic_features_val = np.array(classic_features_val).reshape(1,4)

    scaler_classic, scaler_cnn, pca, label_encoder, svm = load_scaler_pca_svm()

    features = process_data_for_model(classic_features_val, embeddings)
    prob = svm.predict_proba(features)
    prob_index = np.argmax(prob[0] > 0.5)
    confidence = (prob[0][prob_index]) * 100
    prediction = format_classification(prob[0][prob_index])


    # # Statistics by class

    melanoma_p = pd.read_csv('./data/melanoma_percentiles.csv')
    nevo_p = pd.read_csv('./data/nevo_percentiles.csv')

    ### Formatting output
    print(f"La lesión analizada tiene los siguientes parámetros basado en los lineamientos ABCD")
    print(f"Probablemente la lesión es un {prediction} con una confianza del {confidence:.2f}%")
    print("Comparison between melanomas")
    get_features_percentile_analysis(feature_names, classic_features_val, melanoma_p, 'Melanoma')
    print("Comparison between nevos")
    get_features_percentile_analysis(feature_names, classic_features_val, nevo_p, 'Nevo')
    
    


