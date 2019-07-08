#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
def answer_one():
    df = pd.read_csv("data2",sep="\t")
    return df

answer_one()

def answer_two():
    df = answer_one()
    
    yes = np.sum([df['target'] > 0])
    no = np.sum([df['target'] < 1])
    
    data = np.array([no, yes])
    s = pd.Series(data,index=['unstable','stable'])
    
    return s

answer_two()

from sklearn.model_selection import train_test_split
def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf[  ['+', '.', ':', '*'] ]
    y = cancerdf['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X, y

from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 9, p=6)
    knn.fit(X_train, y_train)
    
    return knn

from sklearn.neighbors import KNeighborsClassifier
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    A = answer_five()
    prediction = A.predict(means)

    return prediction

answer_six()

def answer_seven():
    cancerdf = answer_one()
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    prediction =  knn.predict(X_test)
    
    return prediction

answer_seven()

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = (knn.score(X_test, y_test))
    
    return score

answer_eight()

def accuracy_plot():
    import matplotlib.pyplot as plt

    get_ipython().run_line_magic('matplotlib', 'notebook')

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Unstable\nTraining', 'Stable\nTraining', 'Unstable\nTest', 'Stable\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Unstable and Stable Mutations', alpha=0.8)
    
# Uncomment the plotting function to see the visualization, 
# Comment out the plotting function when submitting your notebook for grading

accuracy_plot()


# In[3]:


def searchBestPar():
    bestScore=0
    bestK=-1
    bestWeight=""
 
    # weight==uniform时
    for k in range(1,10):
        clf = KNeighborsClassifier(n_neighbors=k,weights="uniform")
        clf.fit(trainX,trainY)
        scor=clf.score(testX,testY)
        if scor > bestScore:
            bestScore=scor
            bestK=k
            bestWeight="uniform"
 
    # weight==distance时
    for k in range(1,10):
        for p in range(1,7):
            clf=KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
            clf.fit(trainX,trainY)
            scor = clf.score(testX, testY)
            if scor > bestScore:
                bestScore = scor
                bestK = k
                bestWeight = "distance"
 
    print("the best n_neighbors", bestK)
    print("the best weights", bestWeight)
    print("the best p", p)
 
if __name__ == '__main__':
    trainX, testX, trainY, testY = answer_four()
    searchBestPar()


# In[ ]:




