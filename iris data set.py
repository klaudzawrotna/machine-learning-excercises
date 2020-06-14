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


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


# In[40]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[41]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))


# In[44]:


sample  = [[3,5,4,2],[2,3,5,4]]
predictions = knn.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print("predictions: ", pred_species)


# In[47]:


from sklearn.externals import joblib
model = joblib.dump(knn, 'mlbrain.joblib')


# In[49]:


model = joblib.load('mlbrain.joblib')
model.predict(X_test)
sample  = [[3,5,4,2],[2,3,5,4]]
predictions = model.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print("predictions: ", pred_species)


# In[ ]:




