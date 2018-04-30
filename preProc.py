import numpy as np
import pandas as pd
from sklearn import preprocessing


def cleaningByMV(numData, categData, dataSize, threshold=0.3):

    categMV = []
    for i in categData.columns:
        categMV.append(sum(pd.isnull(categData[i])))

    categMV = np.array(categMV)

    numMV = []
    for i in numData.columns:
        numMV.append(sum(pd.isnull(numData[i])))

    numMV = np.array(numMV)

    categData = categData.loc[:, categMV < (dataSize * threshold)]
    numData = numData.loc[:, numMV < (dataSize * threshold)]

    return numData, categData


def cleaningCategories(categData, threshold):

    categCount = []
    for i in categData.columns:
        categCount.append(categData.groupby(i).size().size)
    categCount = np.array(categCount)
    categData = categData.loc[:, categCount < threshold]

    return categData


def fillingNumMV(numData):

    for i in numData.columns:
        meanCol = numData[i].mean()
        mask = numData[i].isnull().tolist()
        numData.ix[mask, i] = meanCol

    return numData


def fillingCategMV(categData):

    for i in categData.columns:
        categData[i] = categData[i].cat.add_categories('missing').fillna('missing')

    return categData


def procData(dataSize, threshMV, threshCateg):

    print('Reading and preparing data...')
    #Reading data
    df = pd.read_csv('data/orange_small_train.data', delimiter='\t') 
    categData = df.iloc[:, 190:230]
    numData = df.iloc[:, 0:190]

    churnLabels = np.loadtxt('data/orange_small_train_churn.labels')
    appetencyLabels = np.loadtxt('data/orange_small_train_appetency.labels')
    upsellingLabels = np.loadtxt('data/orange_small_train_upselling.labels')

    for i in categData.columns:
        x = categData[i].astype('category')
        categData.ix[:, i] = x

    #Cleaning and preparing data

    numData, categData = cleaningByMV(numData, categData, dataSize, threshMV)
    numData = fillingNumMV(numData)
    categData = cleaningCategories(categData, threshCateg)
    categData = fillingCategMV(categData)

    X_c = pd.get_dummies(categData).values
    X_n = numData.values
    X_n = preprocessing.minmax_scale(X_n)
    X = np.hstack((X_n, X_c))
    
    return X, churnLabels, appetencyLabels, upsellingLabels
