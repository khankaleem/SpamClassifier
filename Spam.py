from math import log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import ProcessData

class SpamClassifier(object):
    def __init__(self, X_train):
        self.mails, self.labels = X_train['message'], X_train['label']
        self.Prob_Spam = dict()
        self.Prob_Ham = dict()
        self.TF_Spam = dict()
        self.TF_Ham = dict()
        self.IDF_Spam = dict()
        self.IDF_Ham = dict()
        self.SUM_TF_IDF_Spam = 0
        self.SUM_TF_IDF_Ham = 0        
        self.Spam_Mails = self.labels.value_counts()[1]
        self.Ham_Mails = self.labels.value_counts()[0]
        self.Total_Mails = self.Spam_Mails + self.Ham_Mails
        self.Prob_SpamMail = self.Spam_Mails/self.Total_Mails 
        self.Prob_HamMail = self.Ham_Mails/self.Total_Mails
        
    def Train(self):
        self.Build_TF_IDF()
        self.BuildProbability()
        
    def Build_TF_IDF(self):
        for i in range(self.Total_Mails):
            Msg = ProcessData.Process_Msg(self.mails[i])
            count = list()
            for word in Msg:
                if self.labels[i]:
                    self.TF_Spam[word] = self.TF_Spam.get(word, 0)+1
                else:
                    self.TF_Ham[word] = self.TF_Ham.get(word, 0)+1
                
                if word not in count:
                    count += [word]

            for word in count:
                if self.labels[i]:
                    self.IDF_Spam[word] = self.IDF_Spam.get(word, 0)+1
                else:
                    self.IDF_Ham[word] = self.IDF_Ham.get(word, 0)+1
        
    def BuildProbability(self):
        '''
        calculate P(word|Spam) = (TF(word|Spam)*IDF(word) + 1)/sum(P(x|Spam)) + distinct words in the corresponding Spam
        '''
        for word in self.TF_Spam:
            self.Prob_Spam[word] = self.TF_Spam[word]*log(self.Total_Mails/(self.IDF_Spam.get(word, 0)+self.IDF_Ham.get(word,0)))
            self.SUM_TF_IDF_Spam += self.Prob_Spam[word]
        for word in self.TF_Spam:
            self.Prob_Spam[word] = (self.Prob_Spam[word]+1)/(self.SUM_TF_IDF_Spam + len(list(self.Prob_Spam.keys())))
        
        for word in self.TF_Ham:
            self.Prob_Ham[word] = self.TF_Ham[word]*log(self.Total_Mails/(self.IDF_Spam.get(word, 0)+self.IDF_Ham.get(word,0)))
            self.SUM_TF_IDF_Ham += self.Prob_Ham[word]
        for word in self.TF_Ham:
            self.Prob_Ham[word] = (self.Prob_Ham[word]+1)/(self.SUM_TF_IDF_Ham + len(list(self.Prob_Ham.keys())))

    
    def Classify(self, msg):
        P_spam, P_ham = 0, 0         
        
        for word in msg:
            if word in self.Prob_Spam:
                P_spam += log(self.Prob_Spam[word])
            else:
                P_spam -= log(self.SUM_TF_IDF_Spam + len(list(self.Prob_Spam.keys())))
            
            if word in self.Prob_Ham:
                P_ham += log(self.Prob_Ham[word])
            else:
                P_ham -= log(self.SUM_TF_IDF_Ham + len(list(self.Prob_Ham.keys())))        
        P_spam += log(self.Prob_SpamMail)
        P_ham += log(self.Prob_HamMail)
        return P_spam >= P_ham
        
    def Predict(self, test_data):
        result = dict()
        for (i, message) in enumerate(test_data):
            msg = ProcessData.Process_Msg(message)
            result[i] = int(self.Classify(msg))
        return result    
     
    def Accuracy(self, labels, result):
        correct = 0
        for i in range(len(labels)):
            if labels[i] == result[i]:
                correct += 1
        print('Accuracy: '+str((correct/len(labels))*100.0))