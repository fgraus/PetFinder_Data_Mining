import sys
sys.path.append('./')
import pandas as pd
from PIL import Image
import os

def getExtraInformation():
    extraInformation = pd.read_csv('resources/train.csv', index_col=0)
    extraInformation = getExtraInformationWithMetadata(extraInformation)
    return extraInformation

def getTrainPhotosList():
    photoList = []
    for root, dirs, files in os.walk('.', topdown=False):
        if root == '.\\resources\\train':
            for file in files:
                photoList.append(file)
    return photoList

def getExtraInformationWithMetadata(info):
    info['1440'] = 0
    info['960'] = 0
    info['810'] = 0
    info['720'] = 0
    info['240'] = 0
    for i in range(info.shape[0]):
        image = Image.open('.\\resources\\train\\' + info.iloc[i].name+'.jpg')
        if 1100 <= image.height:
            info.at[info.iloc[i].name, '1440'] = 1
        elif 1100 > image.height and image.height >= 900:
            info.at[info.iloc[i].name, '960'] = 1
        elif 900 > image.height and image.height >= 700:
            info.at[info.iloc[i].name, '810'] = 1
        elif 700 > image.height and image.height >= 600:
            info.at[info.iloc[i].name, '720'] = 1
        elif 600 > image.height:
            info.at[info.iloc[i].name, '240'] = 1

    return info

