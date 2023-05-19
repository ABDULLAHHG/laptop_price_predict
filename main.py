import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as slt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor

# Read dataset 
df = pd.read_csv("~/data/laptop_price.csv",encoding='Latin')

#slt.dataframe(df)
selectdict = {'company':[i for i in df.Company.value_counts().index],
              'CPU_Brand' : [ i[0] for i in df.Cpu.value_counts().index]}

#company =slt.selectbox('Company',[i for i in df.Company.value_counts().index])
#slt.dataframe(df[df['Company'] == company ])
#slt.selectbox('dataframe',selectdict )
#slt.select_slider('company',selectdict)

# Convet CPU to Frequency and Brand  
brand = (df.Cpu.apply( lambda x:x.split()[0])).rename('CPU_Brand')
frequency = ((df.Cpu.apply( lambda x:x.split()[-1].replace("GHz",'')).astype(float)*1000).rename('CPU_Frequency'))

# Create GPU Brand column
GPU_Brand = df.Gpu.apply(lambda x:x.split()[0]).rename('GPU_Brand')


# Convert Screen Resolution into Height and Width 
SR =df.ScreenResolution.apply(lambda x:x.split()[-1])
height = SR.apply(lambda x:x.split('x')[-1])
width = SR.apply(lambda x:x.split('x')[0])
height =(height.rename('Height')).astype(int)
width = (width.rename('Width')).astype(int)
df = df.join([GPU_Brand,brand,frequency,width,height])


# There is laptops has more than 1 hard and less than 3 
df["Memory-1"] = df.Memory.apply(lambda x:x.split('+')[0])
df["Memory-2"] = df.Memory.apply(lambda x:x.split('+')[-1] if "+" in x else '0GB')

# Split types and size of each memory
df['Type-Memory-1'] =df['Memory-1'].apply(lambda x:x.split()[1])
df['Size-Memory-1'] =df['Memory-1'].apply(lambda x:float(x.split()[0].replace('GB',''))*1000 if 'GB' in x else float(x.split()[0].replace('TB',''))*1000000)
df['Type-Memory-2'] =df['Memory-2'].apply(lambda x:x.split()[1] if x != '0GB' else 'No storage')
df['Size-Memory-2'] =df['Memory-2'].apply(lambda x:float(x.split()[0].replace('GB',''))*1000 if 'GB' in x else float(x.split()[0].replace('TB',''))*1000000)

# convert Weight to float
df.Weight = df.Weight.apply(lambda x:float(x.replace('kg',''))*1000)

# convert RAM to MB and convert it to int 
df.Ram = df.Ram.apply(lambda x:int(x.replace('GB',''))*1000)

# Convert string colums to binary (get_dummies)
GPU_Brand = pd.get_dummies(df['GPU_Brand'],prefix = 'GPU').astype(int)
CPU_Brand = pd.get_dummies(df['CPU_Brand'],prefix = 'CPU').astype(int)
company = pd.get_dummies(df['Company']).astype(int)
OpSys = pd.get_dummies(df['OpSys']).astype(int)
TypeName = pd.get_dummies(df['TypeName']).astype(int)
Type_Memory_1 = pd.get_dummies(df['Type-Memory-1'],prefix ='Type_Memory_1').astype(int)
Type_Memory_2 = pd.get_dummies(df['Type-Memory-2'],prefix ='Type_Memory_2').astype(int)

# list of variables for user to input 
list_of_copmanies = df['Company'].value_counts().index
list_of_CPU = df['Cpu'].value_counts().index
list_of_GPU = df['Gpu'].value_counts().index
list_of_Types = df['TypeName'].value_counts().index
list_of_SR = df['ScreenResolution'].value_counts().index
list_of_OpSys = df['OpSys'].value_counts().index
list_of_Memories = df['Memory'].value_counts().index
list_of_Ram = df['Ram'].value_counts().index

# Join columns 
df = df.join([GPU_Brand,CPU_Brand,company,OpSys,TypeName,Type_Memory_1,Type_Memory_2])

# Drop columns 
df = df.drop('ScreenResolution',axis=1)
df = df.drop('Memory-1',axis = 1)
df = df.drop('Memory-2',axis = 1)
df = df.drop('Memory',axis =1 )
df = df.drop('Cpu',axis = 1 )
df = df.drop('Gpu',axis = 1 )
df = df.drop('GPU_Brand',axis = 1 )
df = df.drop('CPU_Brand',axis = 1 )
df = df.drop('Company',axis = 1 )
df = df.drop('OpSys',axis = 1)
df = df.drop('TypeName',axis = 1 )
df = df.drop('Type-Memory-1', axis = 1 )
df = df.drop('Type-Memory-2', axis = 1 )
df = df.drop('Product' ,axis = 1 )
df = df.drop('laptop_ID',axis = 1)

# show dataframe on website 
#slt.dataframe(df)

## Feature selection 
# Specific column with higher corr 
feature = abs(df.corr()).sort_values(by='Price_euros')[-15::].index

# show corrilation 
#plt.figure(figsize = (20,10))
#fig = sns.heatmap(df[feature].corr(),fmt='.2f',annot =True ,cmap='YlGnBu') 
#slt.pyplot(plt)

# Split the data to train and test 
X = df.drop('Price_euros',axis = 1 )
y = df['Price_euros']
X_trian ,X_test ,y_train , y_test = train_test_split(X,y,test_size=0.25)

# scale the data with StandardScaler
SC = StandardScaler()
SC.fit(X_trian)
X_train = SC.transform(X_trian)
X_test = SC.transform(X_test)

## Machaine Learing 
# Random Forest Regressor 
RFR = RandomForestRegressor(n_estimators=10000)
RFR.fit(X_train, y_train)
y_hat = RFR.predict(X_test)

# accuracy check Random Forest Regressor 
MAE = mean_absolute_error(y_test,y_hat)
print(MAE)
MSE = mean_squared_error(y_test,y_hat)
print(MSE)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)

# accuracy check Random Linear Regression  
print('Linear Regression')
MAE = mean_absolute_error(y_test,y_hat)
print(MAE)
MSE = mean_squared_error(y_test,y_hat)
print(MSE)


# Initialize individual regression models
model3 = DecisionTreeRegressor()

# Create a voting ensemble using the individual models
ensemble_model = VotingRegressor([('rf', RFR), ('dt', model3)])

# Fit the ensemble model to your training data
ensemble_model.fit(X_train, y_train)

# Make predictions using the ensemble model
y_hat = ensemble_model.predict(X_test)

# accuracy check Random Linear Regression  
print('ensemble model')
MAE = mean_absolute_error(y_test,y_hat)
print(MAE)
MSE = mean_squared_error(y_test,y_hat)
print(MSE)


## Select box 
# CPU 


# GPU 

# Ram size 

# Memory 

# Screen Resolution 

# Company


