#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


dataset=pd.read_csv("data.csv")


# In[119]:


dataset.head()


# In[120]:


dataset.tail()


# In[121]:


dataset.describe()


# In[122]:


print(dataset.dtypes)


# In[123]:


dataset.duplicated().sum()


# In[124]:


dataset=dataset.drop_duplicates()


# In[125]:


dataset.isnull().sum()


# In[126]:


dataset.info()


# In[127]:


dataset['city'].unique()


# In[128]:


dataset['city'].value_counts()


# In[129]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[130]:


dataset.columns


# In[131]:


dataset['street'].value_counts()


# In[132]:


dataset['yr_built'].value_counts()


# In[133]:


sns.distplot(dataset.price)


# In[134]:


print("Skewness: %f" % dataset['price'].skew())
print("Kurtosis: %f" % dataset['price'].kurt())


# In[135]:


import seaborn as sns
sns.pairplot(dataset)


# In[136]:


dataset['waterfront'].value_counts()


# In[137]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[138]:


dataset['view'].value_counts()


# In[139]:


dataset['city']=pd.factorize(dataset['city'])[0]
dataset['country']=pd.factorize(dataset['city'])[0]
dataset['street']=pd.factorize(dataset['city'])[0]


# In[101]:


dataset.shape


# In[ ]:


dataset.info()


# In[142]:


dataset.head()


# In[72]:


dataset.tail()


# In[166]:


dataset.info()


# In[168]:


dataset.head()


# In[169]:


b=(dataset.columns)


# In[170]:


a=StandardScaler()
dataset=a.fit_transform(dataset)
dataset=pd.DataFrame(dataset,columns=b)


# In[171]:


dataset.head()


# In[ ]:





# In[253]:


X=dataset.drop(['price','city'],axis=1)
y=dataset[['price']]


# In[254]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)


# In[255]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)
lr.score(X_test,y_test)


# In[256]:


#Svm
from sklearn import preprocessing
from sklearn import utils


# In[257]:


lab=preprocessing.LabelEncoder()
yt=lab.fit_transform(y_train)
yt1=lab.fit_transform(y_test)


# In[258]:


print(yt.shape)
print(X_train.shape)


# In[259]:


from sklearn.svm import SVR
svr_rbf=SVR(kernel='rbf')
svr_rbf.fit(X_train,y_train)
print(svr_rbf.score(X_test,y_test))


# In[260]:


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=1)
poly_reg.fit(X_train)
train_poly=poly_reg.transform(X_train)
test_poly=poly_reg.transform(X_test)


# In[261]:


lr.fit(train_poly,y_train)
lr.score(test_poly,y_test)


# In[262]:


#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression 
regr = RandomForestRegressor(max_depth=3, random_state=101)
regr.fit(X_train, y_train)
regr.score(X_test, y_test)


# In[ ]:





# In[ ]:




