import os
import torch
import segmentation_models_pytorch as smp
from torchvision import models 
import torch.nn as nn
import joblib
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

HOME_DIR = Path.home()

MODEL_PATH = os.path.join(BASE_PATH, '.', 'data', 'model_weights_skin_segmentation_ham10000.pth')



device = torch.device('mps')
model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location='mps'))
model.eval()


model_resnet = models.resnet50(weights='DEFAULT').to(device)
model_resnet.fc = nn.Identity()
model_resnet.eval()



def load_scaler_pca_svm():
    scaler_classic = joblib.load('./data/scaler_classic.pkl')
    scaler_cnn = joblib.load('./data/scaler_cnn.pkl')
    pca = joblib.load('./data/pca.pkl')
    svm = joblib.load('./data/model_svm_08_sampling_strgy.pkl')
    label_encoder = joblib.load('./data/label_encoder.pkl')

    return scaler_classic, scaler_cnn, pca, label_encoder,svm