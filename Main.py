import Spam
import LoadData

TrainData, TestData = LoadData.GetData()

classifier = Spam.SpamClassifier(TrainData)
classifier.Train()    
result = classifier.Predict(TestData['message'])
classifier.Accuracy(TestData['label'], result)