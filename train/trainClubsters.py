import sys
sys.path.append('./')
from src.models.clouster import k_means, getAccuracy
from src.handleData import getExtraInformation
import pandas as pd
from tqdm import tqdm

def crossValidation(data, numbersChuncks, numberClousters = 40):

    scoreSum = 0

    for i in range(0,numbersChuncks):
        posStart = i*round(data.shape[0]/numbersChuncks)
        posEnd = (i + 1) * round(data.shape[0]/numbersChuncks)

        test = data.iloc[posStart:posEnd]
        train = data.iloc[:posStart]
        train = train.append(data.iloc[posEnd:])

        model, data = k_means(data, numberClousters)

        scoreSum += getAccuracy(model, data, test)

    print('Total score is ' + str(scoreSum/numbersChuncks) + 'with this amounts of clubters' + str(numberClousters))
    return scoreSum/numbersChuncks

def testCrossValidationValues():
    print('started')
    featureInformation = getExtraInformation()
    a = pd.DataFrame(columns=['clubsters','error'])
    f = open('dataReport/FeaturesClubstersNumberClubster.txt', 'w')
    for i in tqdm(range(2, 200,20)):
      a = a.append({'clubsters' : i, 'error' : crossValidation(featureInformation, 10, i)}, ignore_index=True)

    for i in range(a.shape[0]):
        f.write('numberClubsters ' + str(a.iloc[i].clubsters) + ' error ' + str(a.iloc[i].error) +'\n')
        print('numberClubsters ' + str(a.iloc[i].clubsters) + ' error ' + str(a.iloc[i].error))
    f.close()
#testCrossValidationValues()