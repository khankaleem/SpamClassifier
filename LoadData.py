import pandas as pd
import numpy as np
def GetData():   
    dataset = pd.read_csv('spam.csv', encoding = 'latin-1')
    dataset = dataset.dropna(axis = 1)
    dataset['label'] = dataset['labels'].map({'ham': 0, 'spam': 1})
    dataset.drop(['labels'], axis = 1, inplace = True)
    
    totaldataset = 4825 + 747
    TrainIndex, TestIndex = list(), list()
    for i in range(dataset.shape[0]):
        if np.random.uniform(0, 1) < 0.75:
            TrainIndex += [i]
        else:
            TestIndex += [i]
    TrainData = dataset.loc[TrainIndex]
    TestData = dataset.loc[TestIndex]
    
    TrainData.reset_index(inplace = True)
    TrainData.drop(['index'], axis = 1, inplace = True)
    
    TestData.reset_index(inplace = True)
    TestData.drop(['index'], axis = 1, inplace = True)
    
    return (TrainData, TestData)