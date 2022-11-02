from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import numpy as np

def nearestNeighbords(train, test, numberNeighbourds):
    neighbors = NearestNeighbors(n_neighbors = numberNeighbourds, algorithm='brute', metric='matching').fit(np.stack(train.drop(columns = ['Pawpularity']).values))
    distances, indices = neighbors.kneighbors(test.drop(columns = ['Pawpularity']).values)
    true = np.empty(0)
    pred = np.empty(0)
    for i in range(test.shape[0]):
        true = np.append(true, test.iloc[i].Pawpularity)
        pred = np.append(pred, train.iloc[indices[i]]['Pawpularity'].mean())

    return mean_squared_error(true, pred)

def getnearestNeighbords(train):
    neighbors = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='matching').fit(np.stack(train.drop(columns = ['Pawpularity']).values))
    return neighbors

def getPredictionFeatures(model,train, toPredict):
    distances, indices = model.kneighbors(toPredict.drop(columns = ['Pawpularity']).values)
    toPredict['prePrediction'] = 0
    for i in range(toPredict.shape[0]):
        toPredict.at[toPredict.iloc[i].name, 'prePrediction'] = train.iloc[indices[i]]['Pawpularity'].mean()
    return toPredict