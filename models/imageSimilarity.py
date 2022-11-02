import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from numpy.linalg import norm
from src.handleData import getTrainPhotosList, getExtraInformation

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.densenet import DenseNet201
from tqdm import tqdm

def getSimilarityMatrix(train = False, modelo='xception'):
    if train:
        filenames = getTrainPhotosList()
        photoInfo = getExtraInformation()
        photoInfo['features'] = [np.empty(0, dtype=float)]*len(photoInfo)
        if modelo == 'xception':
            model = Xception(weights='imagenet', include_top=False, input_shape= (320, 240, 3))
        if modelo == 'nnasnetlarge':
            model = NASNetLarge(weights='imagenet', include_top=False, input_shape= (331, 331, 3))
        if modelo == 'densenet':
            model = NASNetLarge(weights='imagenet', include_top=False, input_shape= (320, 240, 3))
        for i in tqdm(range(len(filenames))):
            photoInfo.at[filenames[i].replace('.jpg', ''), 'features'] = extract_features(filenames[i], model)
        photoInfo.to_pickle('resources/pickels/trainPhotoFeatures'+ modelo+'.pkl')
    else:
        photoInfo = pd.read_pickle('resources/pickels/trainPhotoFeatures'+ modelo+'.pkl')
    return photoInfo

def getPredictions(train, test, numberNeighbourds):
    test['prediccion'] = 0
    neighbors = NearestNeighbors(n_neighbors=numberNeighbourds, algorithm='brute', metric='euclidean').fit(np.stack(train.features._values))
    distances, indices = neighbors.kneighbors(np.stack(test.features._values))
    for i in range(test.shape[0]):
        num = 0
        den = 0
        for j in range(distances[i].shape[0]):
            num += distances[i][j] * train.iloc[indices[i][j]]['Pawpularity']
            den += distances[i][j]
        test.at[test.iloc[i].name, 'prediccion'] = num/den
    return test

def getPredictionByNeighbourds(train, test, numberNeighbourds):
    neighbors = NearestNeighbors(n_neighbors=numberNeighbourds, algorithm='brute', metric='euclidean').fit(np.stack(train.features._values))
    distances, indices = neighbors.kneighbors(np.stack(test.features._values))
    true = np.empty(0)
    pred = np.empty(0)
    for i in range(test.shape[0]):
        num = 0
        den = 0
        for j in range(distances[i].shape[0]):
            num += distances[i][j] * train.iloc[indices[i][j]]['Pawpularity']
            den += distances[i][j]
        true = np.append(true, test.iloc[i].Pawpularity)
        pred = np.append(pred, (num/den))
    return mean_squared_error(true, pred)

def extract_features(img_path, model):
    input_shape = (320, 240, 3)
    img = image.load_img('resources/train/' + img_path, target_size=(input_shape[0],input_shape[1]), interpolation='nearest')
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

#getSimilarityMatrix()