from sklearn import svm, metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def load_scaler_pca_svm():
    scaler_classic = joblib.load('./data/scaler_classic.pkl')
    scaler_cnn = joblib.load('./data/scaler_cnn.pkl')
    pca = joblib.load('./data/pca.pkl')
    svm = joblib.load('./data/model_svm_08_sampling_strgy.pkl')
    label_encoder = joblib.load('./data/label_encoder.pkl')

    return scaler_classic, scaler_cnn, pca, label_encoder,svm


scaler_classic, scaler_cnn, pca, label_encoder,svm = load_scaler_pca_svm()

base_path = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(base_path, '..', 'data', 'embeddings.npz')
classic_features_path = os.path.join(base_path, '..', 'data', '03_dataset_w_features.csv')

features_efficient_net = np.load(embeddings_path, allow_pickle=True)

dataset = pd.read_csv(classic_features_path, index_col=0)
dataset = dataset.sort_values('image_id').reset_index(drop=True)
mask = (dataset['total_x'] == 0.0) & (dataset['total_y'] == 0.0)
dataset = dataset[~mask].dropna()
dataset = dataset.drop_duplicates(subset=['image_id'], keep=False, inplace=False, ignore_index=False)
valid_ids = dataset['image_id'].values
mask_ids = np.isin(features_efficient_net['base_ids'], valid_ids)
embeddings_valid = features_efficient_net['embeddings'][mask_ids]
id_embeddings = features_efficient_net['base_ids'][mask_ids]

X_classic = dataset[['diameter', 'circularity', 'total_x', 'total_y']] 
X_cnn = embeddings_valid
y = dataset['dx'].values

X_scaled_classic = scaler_classic.transform(X_classic)


X_scaled_cnn = scaler_cnn.transform(X_cnn)

label_encoder = LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Apply pca over cnn features.

X_cnn_pca = pca.transform(X_scaled_cnn)

X_stacked = np.hstack([X_scaled_classic, X_cnn_pca ])



def get_recommendations(query):

    scores = cosine_similarity(query, X_stacked)[0]


    index = np.argsort(scores)[::-1]
    
    paths_data = dataset.iloc[index]

    return paths_data[:5]

