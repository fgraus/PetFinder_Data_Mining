import sys
sys.path.append('./')
import pandas as pd
from tqdm import tqdm

from src.handleData import getExtraInformation
from src.models.nearestNeighbords import nearestNeighbords

def crossValidation(data, numbersChuncks, numberNeighbords = 40):

    scoreSum = 0

    for i in range(0,numbersChuncks):
        posStart = i*round(data.shape[0]/numbersChuncks)
        posEnd = (i + 1) * round(data.shape[0]/numbersChuncks)

        test = data.iloc[posStart:posEnd]
        train = data.iloc[:posStart]
        train = train.append(data.iloc[posEnd:])

        scoreSum += nearestNeighbords(train, test, numberNeighbords)

    print('Total score is ' + str(scoreSum/numbersChuncks) + 'with this amounts of neighbors' + str(numberNeighbords))
    return scoreSum/numbersChuncks

def testNearestNeighbords():
    featureInformation = getExtraInformation()
    a = pd.DataFrame(columns=['neighbourds','error'])
    f = open('dataReport/FeaturesNeighbordsNumberNeigh.txt', 'w')
    for i in tqdm(range(30, 500, 5)):
      a = a.append({'neighbourds' : i, 'error' : crossValidation(featureInformation, 10, i)}, ignore_index=True)

    for i in range(a.shape[0]):
        f.write('neighbourds ' + str(a.iloc[i].neighbourds) + ' error ' + str(a.iloc[i].error) + '\n')
        print('neighbourds ' + str(a.iloc[i].neighbourds) + ' error ' + str(a.iloc[i].error))
    f.close()

testNearestNeighbords()