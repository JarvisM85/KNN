# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:59:36 2024

@author: sahil
"""

import pandas as pd
import numpy as np
wbcd = pd.read_csv("C:/DS2/1.2_KNN/wbcd.csv")  

wbcd.describe()
# in o\p colm there is only B for Benien and M for Malignant
wbcd['diagnosis'] = np.where(wbcd['diagnosis']=='B','Beniegn',wbcd['diagnosis'])
# In wbcd there is colm named 'diagnosis', where ever there
# is 'B' replace with 'Benign'
# Similar for 'M' with 'Malignant'
wbcd['diagnosis'] = np.where(wbcd['diagnosis']=='M','Malignant',wbcd['diagnosis'])
#############################
# 0th colmm is patent Id , so drop it
wbcd = wbcd.iloc[:,1:32]
#############################
# Normalization
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return x
#Now let us apply this function to the dataframe
wbcd_n = norm_func(wbcd.iloc[:,1:32])
# because now 0 th colm is o\p or label it is 
# not considered hence 

##########
# Let us now apply X as i\p and y as o\p
X = np.array(wbcd_n.iloc[:,:])
# since in wbcd_n , we already excluding o\p colm,
# hence all rows and 
y = np.array(wbcd['diagnosis'])

###########################
# Now let us split the data into training and testing
from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2)

# here you are passing X,y instead dataframe handle
# there could chances of unbalanced data
# let us assume you have 100 data points, with 80 NC & 20 cancer
# This data point must be equally distrbuted
#There is "Stratified Sampling" Concept is used
#        ----------------------   

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred

# Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)


######################

acc = []

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc = np.mean(neigh.predict(X_train)==y_train)
    test_acc = np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)

