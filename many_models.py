#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
df = pd.read_csv("data4",sep="\t")
df.head()
print(df.describe())
print(df[df['target']=='unstable'].describe())
print(df[df['target']=='stable'].describe())
print(df[df['target']=='unchanged'].describe())


# In[27]:


df.plot(kind = 'box', subplots = True, layout = (4, 4), sharex = False, sharey = False)
plt.show()


# In[28]:


df.hist()
his = plt.gcf()
his.set_size_inches(12, 6)
plt.show()


# In[29]:


sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'target', size = 6).map(plt.scatter, ':', '*').add_legend()
plt.show()


# In[30]:


plt.close()
sns.pairplot(df, hue = 'target', height = 2, diag_kind = 'kde')
plt.show()


# In[31]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='target',y='+',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='target',y='.',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='target',y=':',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='target',y='*',data=df)

plt.show()


# In[32]:


# Import modules
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


# In[33]:


df.head()


# In[34]:


plt.figure(figsize=(7,5)) 
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn_r') 
plt.show()


# In[35]:


#spliting the data
test_size = 0.30
seed = 7
score = 'accuracy'
# Implementation of different ML Algorithms
def models(X_train, Y_train,score):
    clfs = []
    result = []
    names = []
    clfs.append(('LR', LogisticRegression()))
    clfs.append(('LDA', LinearDiscriminantAnalysis()))
    clfs.append(('KNN', KNeighborsClassifier()))
    clfs.append(('CART', DecisionTreeClassifier()))
    clfs.append(('NB', GaussianNB()))
    clfs.append(('SVM', SVC()))
    for algo_name, clf in clfs:
        k_fold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_score = model_selection.cross_val_score(clf, X_train, Y_train, cv=k_fold, scoring=score)
        #result = "%s: %f (%f)" % (algo_name, cv_score.mean(), cv_score.std())
        result.append((algo_name,cv_score.mean(), cv_score.std()))
        names.append(algo_name)
    return (result)


# In[36]:


X_all = df.iloc[:,:4]
Y_all = df.iloc[:,4] 


# In[37]:


X_train_all, X_test_all, Y_train_all, Y_test_all = model_selection.train_test_split(X_all, Y_all, test_size=test_size, random_state=14)


# In[38]:


models(X_train_all, Y_train_all, score)


# In[39]:


# Evaluation of the Classifier 
# Predictions on test dataset
svm = SVC()
svm.fit(X_train_all, Y_train_all)
pred = svm.predict(X_test_all)
print(accuracy_score(Y_test_all, pred))
print(confusion_matrix(Y_test_all, pred))
print(classification_report(Y_test_all, pred))


# In[19]:


X_sep = df[['*','.']]
Y_sep = df.target


# In[20]:


X_train_sep, X_test_sep, Y_train_sep, Y_test_sep = model_selection.train_test_split(X_sep, Y_sep, test_size=test_size, random_state=seed)
models(X_train_sep, Y_train_sep, score)


# In[22]:


svm = SVC()
svm.fit(X_train_sep, Y_train_sep)
pred = svm.predict(X_test_sep)
print(accuracy_score(Y_test_sep, pred))
print(confusion_matrix(Y_test_sep, pred))
print(classification_report(Y_test_sep, pred))


# In[23]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[24]:


a = pd.read_csv('data4',sep="\t", header = None)
i = pd.DataFrame(a)
mut = i.values


# In[104]:


print(mut)
X = mut[1:, 0:4].astype(float)
print(X)
Y = mut[1:, 4]
print(Y)


# In[25]:


X[0:5]


# In[106]:


Y[0:5]


# In[131]:


from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1]
y=df.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y)
print(x_train,x_test,y_train,y_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)


# In[114]:


mlp = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
mlp.fit(x_train, y_train) 


# In[115]:


print(mlp.score(x_test,y_test))


# In[116]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(x_train, y_train)


# In[117]:


# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# In[118]:


y_true, y_pred = y_test , clf.predict(x_test)
print(set(y_test) - set(y_pred))
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_test, y_pred))


# In[133]:


svm = SVC()
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
print(set(y_test) - set(y_pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:




