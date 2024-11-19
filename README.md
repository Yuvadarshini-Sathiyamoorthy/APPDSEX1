# APPDSEX1
Implementing Data Preprocessing and Data Analysis

## AIM:
To implement Data analysis and data preprocessing using a data set

## ALGORITHM:
Step 1: Import the data set necessary

Step 2: Perform Data Cleaning process by analyzing sum of Null values in each column a dataset.

Step 3: Perform Categorical data analysis.

Step 4: Use Sklearn tool from python to perform data preprocessing such as encoding and scaling.

Step 5: Implement Quantile transfomer to make the column value more normalized.

Step 6: Analyzing the dataset using visualizing tools form matplot library or seaborn.

## CODING AND OUTPUT:
```
# importing libraries 
import pandas as pd 
import scipy as sc 
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('/content/Toyota.csv') 
print(df.head())
```
![image](https://github.com/user-attachments/assets/6d22d8b9-0550-481e-afaf-82ca217fbeb6)
```
df.shape
df.info()
```
![image](https://github.com/user-attachments/assets/03353f6f-22dc-4a93-a766-ce5cd815285d)
```
df.isna().sum()
```
![image](https://github.com/user-attachments/assets/d9cc618e-1152-4246-93c5-b54031c41da0)
```
df.dropna(axis=0,how='any',inplace=True)
df.isna().sum()
```
![image](https://github.com/user-attachments/assets/61e5ab5f-d84e-4364-9ff3-9dbe3170e09d)
```
df.shape
df.describe()
```
![image](https://github.com/user-attachments/assets/56e35627-4d3e-4803-acf7-735402cfc6d4)
```
# Identify the quartiles
q1, q3= np.percentile (df[ 'Age'], [25, 75])
# Calculate the interquartile range
iqr = q3 - q1
# Calculate the lower and upper bounds 
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr) # Drop the outliers
clean_data = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
df.shape
df
```
![image](https://github.com/user-attachments/assets/b2871ad3-7aee-4539-b7bd-ceb1cb59efe3)
```
df.columns
df.nunique()
df["FuelType"].value_counts()
```
![image](https://github.com/user-attachments/assets/90fabe83-a9df-430e-9392-60d74bed8e59)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=["Petrol","Diesel","CNG"]
e1=OrdinalEncoder(categories=[pm])
df["FuelType"]=e1.fit_transform(df[["FuelType"]])
df
```
![image](https://github.com/user-attachments/assets/9060ad97-4a59-4a9b-a7ae-667b6591f010)
```
le=LabelEncoder()
dfc=df.copy()
dfc
```
![image](https://github.com/user-attachments/assets/61599acb-ef67-4ef5-a75a-6b2515f9f73d)
```
dfc['FuelType']=le.fit_transform(dfc['FuelType'])
dfc
```
![image](https://github.com/user-attachments/assets/66c2b87f-f849-4b28-898b-22151ea61cab)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
enc=pd.DataFrame(ohe.fit_transform(df[['FuelType']]))
enc
```
![image](https://github.com/user-attachments/assets/496c6f0e-9c92-4169-9901-c53b40ad407a)
```
df['FuelType'].value_counts()
from sklearn.preprocessing import MinMaxScaler
df
```
![image](https://github.com/user-attachments/assets/7f7bcc4d-4539-4ec3-a6d9-54eeec3cd0a7)
```
scaler = MinMaxScaler()
df[['Weight']] = scaler.fit_transform(df[['Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/324324f4-a397-41cc-ac57-a68f7ffa4f25)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
dfs = pd.read_csv('/content/Toyota.csv')
dfs.dropna(inplace=True)
dfs[['Age']] = sc.fit_transform(dfs[['Age']])
dfs.head(10)
```

![image](https://github.com/user-attachments/assets/299b69d3-72b0-4ed0-90ef-46872d3a5b40)
```
dfs[['Age']]
```

![image](https://github.com/user-attachments/assets/54aaac23-d870-45a1-8ec6-7f2ce0df4918)
```
dfs[['Age']].skew()
import statsmodels.api as sm
sm.qqplot(dfs['Age'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/56777b50-b071-46e5-ae5d-d4fb52c79196)

## RESULT:
Thus Data analysis and Data preprocessing implemeted using a dataset.
