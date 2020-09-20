import numpy as np
import pandas as pd
from Knn import KNN
import random
data = pd.read_csv('F:/BreastCancerClassifier/BreastCancerClassifier/DataSets/breast-cancer-wisconsin.data')
data.replace('?',-99999,inplace = True)
data.drop(['ID'],1,inplace = True)
full_data = data.astype(float).values.tolist()
random.shuffle(full_data)
test_size = 0.2
train_size = 0.8
train_set = {2.0 : [] , 4.0 : []}
test_set =  {2.0 : [] , 4.0 : []}

train = full_data[:int(len(full_data) * train_size)]
test = full_data[int(len(full_data) * train_size):]


for i in train :
    train_set[i[-1]].append(i[:-1])

for i in test :
    test_set[i[-1]].append(i[:-1])

correct =0
total =0
for dic in test_set :
    for tested in test_set[dic]:
        vote = KNN(train_set,tested,k=10)
        if vote == dic:
            correct +=1
        total += 1    
print(correct/total)        



