import pandas as pd
import re
import string
import numpy as np

class NaiveBayesSpamFilter():
    def __init__(self):
        self.dict_SpamOrNot = {'SPAM': {}, 'NOT_SPAM': {}}
        self.pA = 0
        self.notpA = 0
        self.data = pd.read_csv('spam_or_not_spam.csv').dropna()  
        
    def preprocessing(self, body):
        body = ''.join([char for char in body if char not in string.punctuation])
        for r in ((r'\d', 'number '), (r'\b\w{1,3}\b ', '')):
            body = re.sub(*r, body.lower())
        return body

    def calculate_word_frequencies(self, body, label):
        body = self.preprocessing(body)
        for word in body.split():
            if word not in self.dict_SpamOrNot[label]:
                self.dict_SpamOrNot[label][word] = 1
            else:
                self.dict_SpamOrNot[label][word] += 1
        if label == 'SPAM':
            self.pA += 1 
        else:
            self.notpA += 1

    def train(self, X=None, y=None):
        if X is None and y is None:
            X = self.data['email']
            y = self.data['label'].apply(lambda x: 'SPAM' if x == 1 else 'NOT_SPAM')
        else:
            raise TypeError('Missing 1 required positional argument!')
        for i in X.index:
            body = X.loc[i]
            label = y.loc[i]
            self.calculate_word_frequencies(body, label)
    #     return dict_SpamOrNot

    def calculate_P_Bi(self,word, label): 
        V = len(set(self.dict_SpamOrNot['SPAM'].keys()) | set(self.dict_SpamOrNot['NOT_SPAM'].keys()))
        if word in self.dict_SpamOrNot[label]:
            PBi = (self.dict_SpamOrNot[label][word]+1)/(V + sum(self.dict_SpamOrNot[label].values()))
        else:
            PBi = 1/(V + sum(self.dict_SpamOrNot[label].values()))
        return PBi

    def calculate_P_B(self,text, label):
        PB = 0
        for word in text.split():
            PB += np.log(self.calculate_P_Bi(word, label))
        return PB

    def classify(self,email):
        email = self.preprocessing(email)
        PBA = self.calculate_P_B(email, 'SPAM')
        PBnotA = self.calculate_P_B(email, 'NOT_SPAM')
        PA = self.pA/(self.pA + self.notpA)
        PnotA = self.notpA/(self.pA + self.notpA)
        if (PBA + np.log(PA)) > (PBnotA + np.log(PnotA)):
            result = 'SPAM'
        else:
            result = 'NOT_SPAM'
        return result
