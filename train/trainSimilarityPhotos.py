import sys
sys.path.append('./')
from tqdm import tqdm
import numpy as np
from src.models.imageSimilarity import getSimilarityMatrix, getSimilarityMatrix2, getPredictionByNeighbourds
from sklearn.metrics import mean_squared_error

def findBestPretrainedModel(data, numcherChunks = 10, numberNeighbords = 10):
  scoreSum = 0

  for i in range(numcherChunks):
      if i >= 3:
        break
      posStart = i*round(data.shape[0]/numcherChunks)
      posEnd = (i + 1) * round(data.shape[0]/numcherChunks)

      tests = data.iloc[posStart:posEnd]
      train = data.iloc[:posStart]
      train = train.append(data.iloc[posEnd:])

      scoreSum += getPredictionByNeighbourds(train, tests, numberNeighbords)

  return scoreSum/3

def main(model = 'xception'):
  file = open('dataReport/ImageSimilarity_euclidean_neighbords_'+model+'.txt','w')
  data = getSimilarityMatrix(modelo=model)
  for i in tqdm(range(5,500,25)):
    file.write('number neighbords ' + str(i) + ' accuracy ' + str(findBestPretrainedModel(data,numberNeighbords=i)) + '\n')
  file.close()
  return

#main()
