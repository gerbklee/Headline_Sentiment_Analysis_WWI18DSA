# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import numpy as np
import csv
import math
import pandas as pd



df = pd.read_excel("HeadlineData.xlsx")



df = df.dropna()
print(df.head)



df["label"] = df["label"].str.lower()
df["headline"] = df["headline"].str.lower()
print(df.head)



df['headline'] = df['headline'].str.replace(',', '')
df['headline'] = df['headline'].str.replace(':', '')
df['headline'] = df['headline'].str.replace('.', '')
df['headline'] = df['headline'].str.replace(';', '')
df['headline'] = df['headline'].str.replace("'", '')
df['headline'] = df['headline'].str.replace('?', '')
df['headline'] = df['headline'].str.replace('!', '')
df['headline'] = df['headline'].str.replace('*', '')
df['headline'] = df['headline'].str.replace('(', '')
df['headline'] = df['headline'].str.replace(')', '')
df['headline'] = df['headline'].str.replace('/', '')
df['headline'] = df['headline'].str.replace("`", '')
df['headline'] = df['headline'].str.replace("´", '')
df['headline'] = df['headline'].str.replace("‘", '')
df['headline'] = df['headline'].str.replace("’", '')
df['headline'] = df['headline'].str.replace('\d+', '')
print(df.head)



df["headline"] = df["headline"].str.split(pat=' ')



print(df.head)



dfTraining=df[:1600]
dfTesting=df[1600:]



print(dfTraining.head)
print(dfTesting.head)



def train_naive_bayes(dfTraining):
    for i in dfTraining["headline"]:
        for word in i:
            if word in Vocabulary:
                continue
            else:
                Vocabulary.append(word)

    TrainingDocs = len(dfTraining["headline"])
    VSize = len(Vocabulary)
    print("Training with ",TrainingDocs," Headlines and ", VSize," unique Words")

    TrainingPos = dfTraining[dfTraining["label"]=="p"].count()
    TrainingNeg = dfTraining[dfTraining["label"]=="n"].count()
    TrainingNeu = dfTraining[dfTraining["label"]=="o"].count()
    print("Positive: ",TrainingPos["label"])
    print("Negative: ",TrainingNeg["label"])
    print("Neutral: ",TrainingNeu["label"])

    PriorPos = np.log(TrainingPos["label"]/TrainingDocs)
    PriorNeg = np.log(TrainingNeg["label"]/TrainingDocs)
    PriorNeu = np.log(TrainingNeu["label"]/TrainingDocs)

    #for Vword in Vocabulary:
        #for line in TrainingPos["headline"]:
            #for Element in line:
    print(PriorNeu)
    print(PriorNeg)
    print(PriorPos)

   
    



Vocabulary =[]
print(dfTraining["headline"].str[0])



#train_naive_bayes(dfTraining)



class dataObject:
  def __init__(self, data, label):
    self.data = data
    self.label = label



training = []
for index, row in dfTraining.iterrows():
    temp = dataObject(row['headline'], row['label'])
    training.append(temp)
classes = ['p', 'n', 'o']



print(training)



def train_naive_bayes_sepp(training, classes):
    """Given a training dataset and the classes that categorize
    each observation, return V: a vocabulary of unique words,
    logprior: a list of P(c), and loglikelihood: a list of P(fi|c)s
    for each word
    """
    

    #Initialize D_c[ci]: a list of all documents of class i
    #E.g. D_c[1] is a list of [reviews, ratings] of class 1
    D_c = [[]] * len(classes)

    #Initialize n_c[ci]: number of documents of class i
    n_c = [None] * len(classes)

    #Initialize logprior[ci]: stores the prior probability for class i
    logprior = [None] * len(classes)

    #Initialize loglikelihood: loglikelihood[ci][wi] stores the likelihood probability for wi given class i
    loglikelihood = [None] * len(classes)

    

    #Partition documents into classes. D_c[0]: negative docs, D_c[1]: positive docs
    for obs in training:    #obs: a [review, rating] pair
        #if rating >= 90, classify the review as positive
        if obs.label == 'p':
            print('case p')
            print(D_c[0])
            print(obs.data)
            D_c[0] = D_c[0].append(obs.data)    #Can also write as D_c[1] = D_c[1].append(obs)
        #else, classify review as negative
        elif obs.label == 'n':
            print('case n')
            print(D_c[1])
            print(obs.data)
            D_c[1] = D_c[1].append(obs.data)
        elif obs.label == 'o':
            print('case o')
            print(D_c[2])
            print(obs.data)
            D_c[2] = D_c[2].append(obs.data)

    #print(D_c[0])

    return D_c



array = train_naive_bayes_sepp(training, classes)



count = 0
for i in array[2]:
    print(i)
    count += 1
    if count == 5:
        break






