import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as slt

df = pd.read_csv("~/data/laptop_price.csv",encoding='Latin')


selectdict = {'company':[i for i in df.Company.value_counts().index],
              'CPU_Brand' : [ i[0] for i in df.Cpu.value_counts().index]}

#company =slt.selectbox('Company',[i for i in df.Company.value_counts().index])
#slt.dataframe(df[df['Company'] == company ])
#slt.selectbox('dataframe',selectdict )
#slt.select_slider('company',selectdict)

brand = df.Cpu.apply( lambda x:x.split()[0])
frequency = df.Cpu.apply( lambda x:x.split()[-1])
SR =df.ScreenResolution.apply(lambda x:x.split()[-1])
height = SR.apply(lambda x:x.split(x)[-1])
width = SR.apply(lambda x:x.split(x)[1])

df = df.drop('ScreenResolution',axis=1)
df = df.join(SR)
slt.dataframe(df)

