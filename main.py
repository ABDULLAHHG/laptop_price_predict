import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as slt

df = pd.read_csv("~/data/laptop_price.csv",encoding='Latin')
slt.dataframe(df)

selectdict = {'company':[i for i in df.Company.value_counts().index],
              'CPU_Brand' : [ i[0] for i in df.Cpu.value_counts().index]}

company =slt.selectbox('Company',[i for i in df.Company.value_counts().index])
slt.dataframe(df[df['Company'] == company ])
slt.selectbox('dataframe',selectdict )
slt.select_slider('company',selectdict)