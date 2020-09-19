import numpy as np
import pandas as pd
from sklearn import preprocessing ,neighbors
from sklearn.model_selection import train_test_split
def Get_predecit(prediction):
    
    
    for pre in prediction :
        
        if pre == 2:
            print("Maligant")
        else :
            print("NON_Maligant")

data = pd.read_csv('F:/BreastCancerClassifier/BreastCancerClassifier/DataSets/breast-cancer-wisconsin.data')
data.replace('?',-99999,inplace = True)
data.drop(['ID'],1,inplace = True)
y = np.array(data['class'])
x = np.array(data.drop(['class'],1))
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

# accuracy = clf.score(x_test,y_test)
# print(accuracy)
pre = clf.predict(x_test[:10])
Get_predecit(pre)

