import sys
sys.path.append('./')

from src.handleData import getExtraInformation
from src.models.nearestNeighbords import getnearestNeighbords, getPredictionFeatures
from src.models.imageSimilarity import getSimilarityMatrix, getPredictions
from sklearn.metrics import mean_squared_error

def main():

    trainFeatures = getExtraInformation()
    trainImages = getSimilarityMatrix(train=False, modelo='EfficientNetB7')

    testFeatures = trainFeatures.iloc[7000:]
    testImages = trainImages.iloc[7000:]

    trainFeatures = trainFeatures.iloc[:7000]
    trainImages = trainImages.iloc[:7000]

    testImages = getPredictions(trainImages, testImages, 75)
    testFeatures = getPredictionFeatures(getnearestNeighbords(trainFeatures), trainFeatures, testFeatures)
    
    testImages['finalPrediction'] = 0
    for i in range(testImages.shape[0]):
        testImages.at[testImages.iloc[i].name, 'finalPrediction'] = (testImages.iloc[i].prediccion  + testFeatures.iloc[i].prePrediction)/2

    finalScore = mean_squared_error(testImages.Pawpularity, testImages.finalPrediction)

    print(finalScore)


    return

main()