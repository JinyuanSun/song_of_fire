#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  


# In[2]:


df = pd.read_csv("data4",sep="\t")
df.head()


# sns.pairplot(data=df, hue='target', palette='Set2')

# In[42]:


from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1].values
y=df.iloc[:,4].values
x=np.concatenate([x,np.ones(x.shape[0]).reshape(-1,1)],axis = 1) # add a bias vector
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)
print(df['target'].value_counts())


# In[57]:


from sklearn.svm import SVC
def svm_cross_validation(train_x, train_y):    
    model = SVC(class_weight='balanced')    
    param_grid = {'kernel': ['rbf','linear','poly','sigmoid'],
                  'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 
                  'gamma': [1.0,0.1,0.01,0.001, 0.0001],
                  'probability':[True,False]}    
    grid_search = GridSearchCV(model, param_grid, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    
    return model


# In[60]:


from sklearn.svm import LinearSVC
def linearSVC_cross_validation(train_x, train_y):    
    model = LinearSVC(class_weight='balanced')    
    param_grid = {'loss':['hinge','squared_hinge'],'C':[0.01,0.1,0.5,1.0,2.0,10,20]}    
    grid_search = GridSearchCV(model, param_grid, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = LinearSVC(C=best_parameters['C'], loss = best_parameters['loss'],max_iter = 5000)    
    model.fit(train_x, train_y)  
    print("Train ACC: ", model.score(train_x,train_y))
    return model


# In[61]:


model = linearSVC_cross_validation(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))


# print(classification_report(y_test, pred))

# In[68]:


# Select only some of data
# 因为这边 unchanged 的label太多了，导致SVC在predict时候把所有的都认为是unchanged，所以去掉了一部分unchange，让label变得平衡一些。
# 结果如下，确实，当两个class更加平衡的时候，能处理的好一些
data = df[df.target != 'unstable'][:-30]
print(data['target'].value_counts())


# In[69]:


x=data.iloc[:,:-1].values
y=data.iloc[:,4].values
x=np.concatenate([x,np.ones(x.shape[0]).reshape(-1,1)],axis = 1) # add a bias vector
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)


# In[70]:


model = svm_cross_validation(x_train,y_train)


# In[71]:


pred=model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))


# In[73]:


model = linearSVC_cross_validation(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))


# # Try Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=5, max_depth=2,
                             random_state=24)
clf.fit(x_train,y_train)
print("Train Acc: ",clf.score(x_train,y_train))
print("Test Acc: ", clf.score(x_test,y_test))


# In[ ]:




