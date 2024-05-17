#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# ## Loading the Dataset

# In[89]:


import pandas as pd
data=pd.read_csv("Titanic_Dataset.csv")


# In[90]:


data.head()


# In[91]:


data.info()


# ## Preprocessing the Dataset

# In[92]:


from sklearn import preprocessing as pp


# ###  Scaling the Fare Attribute

# In[93]:


data['Fare'].head()


# In[94]:


data_scaler=pp.MinMaxScaler(feature_range=(0,1))
fare_arr=data[['Fare']]
fare_arr


# In[95]:


fare_scaled=data_scaler.fit_transform(fare_arr)


# In[ ]:


fare_scaled


# ### Adding the scaled data as a separate Attribute

# In[97]:


data['fare_scaled']=fare_scaled
data.info()


# In[98]:


col_to_drop=['Name','Cabin']
data=data.drop(col_to_drop,axis=1)
data.info()


# ###  Dropping the Null values

# In[ ]:


data=data.dropna()
data.info()


# In[100]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


# ### Transformin the data from Categorical to numerical

# In[101]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])

print(data['Sex'].unique())


# In[102]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

print(data['Embarked'].unique()) 


# In[103]:


data.head()


# ## Model Implementation

# In[104]:


x=data.drop(['Survived','Ticket','Fare','PassengerId'],axis=1)
y=data['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape


# In[105]:


x.head()


# In[106]:


models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'BN': GaussianNB(),
    'RBF': RandomForestClassifier(),
    'DT': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2*tp)/(2*tp + fp + fn)
    precision = tp/(tp+fp)

    results[name] = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity, #recall
        'Specificity': specificity,
        'F1 Score': f1,
        'Precision': precision,
        
    }

# Print the results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['Accuracy']}")
    print(f"Precision: {result['Precision']}")
    print(f"Sensitivity: {result['Sensitivity']}")
    print(f"Specificity: {result['Specificity']}")
    print(f"F1 Score: {result['F1 Score']}")
    print()


# In[107]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier

models = {
    'ET': ExtraTreesClassifier(),  
    'LIGHTGBM': LGBMClassifier(),  
    'RC': RidgeClassifier(),       
     'LR': LogisticRegression(),
    'gb': GradientBoostingClassifier(),
    'catboost': CatBoostClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2*tp)/(2*tp + fp + fn)
    precision = tp/(tp+fp)

    results[name] = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'Precision': precision,

    }
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['Accuracy']}")
    print(f"Precision: {result['Precision']}")
    print(f"Sensitivity: {result['Sensitivity']}")
    print(f"Specificity: {result['Specificity']}")
    print(f"F1 Score: {result['F1 Score']}")
    
    print()



    


# ## Prediction

# In[108]:


new_passenger = [[1,0,35.0,1,0,2,0.103644]]
model=KNeighborsClassifier()
model.fit(X_train,y_train)
model.predict(X_test)
prediction = model.predict(new_passenger)
print("Survived Prediction:", prediction[0])


# Overll Gradient boosting algorithm gave the highest accuracy of 83.91%.
# 

# In[ ]:




