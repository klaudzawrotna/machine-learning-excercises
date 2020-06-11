#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[9]:


X = iris.data
y = iris.target

feature_names = iris.feature_names
target_name = iris.target_names


# In[8]:


type(X)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


# In[25]:


from sklearn.tree import  DecisionTreeClassifier
knn = DecisionTreeClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[26]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:




