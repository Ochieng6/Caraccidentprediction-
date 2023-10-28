#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy 
from copy import deepcopy
import seaborn as sns 
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk("C:/Users/omond/Downloads/RTA Dataset.csv"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[9]:


rta_dataset = "C:/Users/omond/Downloads/RTA Dataset.csv"
rta_dataset


# In[10]:


# Reading Dataset using pandas
rta_data = pd.read_csv(rta_dataset)

# Safe to keep copies of data
rta_data1 = deepcopy(rta_data)
rta_data2 = deepcopy(rta_data)
rta_data.head()


# In[11]:


#Shape of Dataset
print(f"RTA Dataset has {rta_data.shape[0]} occurences and {rta_data.shape[1]} features!")


# In[12]:


#Columns of Dataset
print("The features of RTA Dataset are:")

features = len(rta_data.columns)
features_list = [feature for feature in rta_data.columns]

for feature in features_list:
    print(feature)


# In[13]:


#Summarized information of columns in Dataset
rta_data.info()


# In[14]:


#Visualization of distribution of Target Variable
rta_data['Accident_severity'].value_counts()


# In[15]:


### Distribution of values

s = sns.countplot(x = 'Accident_severity',data = rta_data)
sizes=[]
for p in s.patches:
    height = p.get_height()
    sizes.append(height)
    s.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(rta_data)*100),
            ha="center", fontsize=14) 


# In[ ]:





# In[9]:


import pandas as pd  # Import the pandas library

# Load your data into the 'rta_data' DataFrame
# Replace 'your_data.csv' with the actual path to your data file
rta_data = pd.read_csv('C:/Users/omond/Downloads/RTA Dataset.csv')

# Now you can use the 'rta_data' DataFrame in your code
# Import missingno and define your function as shown before

import missingno as msno

missingCols = []
nonmissingCols = []

def missingFeatures(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missingCols.append((col, df[col].isnull().sum()))
        else:
            nonmissingCols.append(col)

missingFeatures(rta_data)

missingCols.sort(key=lambda x: x[1], reverse=True)
nonmissingCols.sort()

cols = []

for col, count in missingCols:
    cols.append(col)
cols += nonmissingCols

print(cols)

msno.bar(rta_data[cols])


# In[10]:


msno.matrix(rta_data[cols])


# In[11]:


msno.dendrogram(rta_data[cols])


# In[12]:


rta_data.select_dtypes(include=['object']).head()

