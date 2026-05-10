import os
import torch
import segmentation_models_pytorch as smp
import joblib
from pathlib import Path
from torchvision import models 
import torch.nn as nn

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

HOME_DIR = Path.home()

MODEL_PATH = os.path.join(BASE_PATH, '.', 'data', 'model_weights_skin_segmentation_ham10000_validated.pth')

MODEL_EMBEDDINGS_PATH = f'{BASE_PATH}/data/embeddings_model.pth'



device = torch.device('mps')

model_seg_exists = os.path.exists(MODEL_PATH)
model_emb_exists = os.path.exists(MODEL_EMBEDDINGS_PATH)

model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1).to(device)

if model_seg_exists:
    print('Loading weights for segmentation model')
    model.load_state_dict(torch.load(MODEL_PATH, map_location='mps'))

    
model.eval()


model_resnet = models.resnet50(weights='DEFAULT').to(device)

if model_emb_exists:
    print('Loading weights for embeddings')
    model_resnet.fc = nn.Linear(2048, 64).to(device)
    model_resnet.load_state_dict(torch.load(f'{BASE_PATH}/data/embeddings_model.pth', map_location='mps'))


model_resnet.eval()




def load_scaler_pca_svm():
    scaler_classic = joblib.load(f'{BASE_PATH}/data/scaler_classic.pkl')
    scaler_cnn = joblib.load(f'{BASE_PATH}/data/scaler_cnn.pkl')
    svm = joblib.load(f'{BASE_PATH}/data/model_svm_final.pkl')
    label_encoder = joblib.load(f'{BASE_PATH}/data/label_encoder.pkl')

    return scaler_classic, scaler_cnn, label_encoder,svm