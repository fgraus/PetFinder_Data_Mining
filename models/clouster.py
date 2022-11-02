import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

def k_means(data, number_clouster = 40):
    
    dataWithoutRating = data.drop(columns = ['Pawpularity'])
    
    ms = KMeans(n_clusters = number_clouster, init='k-means++', n_init = 10).fit(dataWithoutRating)

    data['centers'] = ms.labels_

    return ms, data

def getAccuracy(ms, train, test):
    test = test.assign(centers = ms.fit_predict(test.drop(columns = ['Pawpularity']).values))
    for i in range(test.shape[0]):
        test['predicted'] = train.loc[train['centers'] == test.values[i,13]]['Pawpularity'].mean()

    return mean_squared_error(test['Pawpularity'].values, test['predicted'].values)

