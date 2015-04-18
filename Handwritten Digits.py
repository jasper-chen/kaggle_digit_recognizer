
# coding: utf-8

# In[44]:

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas import DataFrame
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
#Random Forest Classifier for Classification problem
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train_labels = train['label'] #(42000,)
train_features = train.drop('label', 1)


# In[70]:

features_train, features_test, labels_train, labels_test = train_test_split(train_features,train_labels, test_size = .33)
print features_train.shape, features_test.shape, labels_train.shape, labels_test.shape 
#(28140,) (13860,) (28140, 784) (13860, 784)
est = RandomForestClassifier(n_estimators=100)
print labels_train
param_grid = {
             'max_depth': [5,10, 15, 20, 25, 30, 35, 40],
             'min_samples_leaf': [2, 3, 5, 10, 20, 40],
}
gs_cv = GridSearchCV(est, param_grid, n_jobs = 4).fit(features_train,labels_train)
gs_cv.fit(features_train,labels_train)


# In[71]:

gs_cv.best_params_


# In[72]:

from sklearn.metrics import accuracy_score
pred = gs_cv.predict(features_test)
accuracy_score = accuracy_score(pred,labels_test)
print accuracy_score
#0.95657


# In[73]:

result = gs_cv.predict(test)
print result.shape
image_id = range(1,len(test)+1)
print len(image_id)

df=pd.DataFrame({'ImageId':image_id, 'Label':result})
df.to_csv('results1.csv', index = False, columns=['ImageId','Label'])


# In[ ]:



