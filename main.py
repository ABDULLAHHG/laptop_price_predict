import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as slt

# Read dataset 
df = pd.read_csv("~/data/laptop_price.csv",encoding='Latin')


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
GPU_Brand = pd.get_dummies(df['GPU_Brand'],prefix = 'GPU')
CPU_Brand = pd.get_dummies(df['CPU_Brand'],prefix = 'CPU')
company = pd.get_dummies(df['Company'])
OpSys = pd.get_dummies(df['OpSys'])
TypeName = pd.get_dummies(df['TypeName'])
Type_Memory_1 = pd.get_dummies(df['Type-Memory-1'],prefix ='Type_Memory_1')
Type_Memory_2 = pd.get_dummies(df['Type-Memory-2'],prefix ='Type_Memory_2')

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
slt.dataframe(df.corr())

## Feature selection 
# Specific column with higher corr 
feature = abs(df.corr()).sort_values(by='Price_euros')[-15::].index

# show corrilation 
plt.figure(figsize = (20,10))
fig = sns.heatmap(df[feature].corr(),fmt='.2f',annot =True ,cmap='YlGnBu') 
slt.pyplot(plt)





