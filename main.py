import pandas as pd 
import streamlit as slt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selectionl import 

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV 

# Header 
slt.header("Laptop Price Predict")

# Read dataset 
df = pd.read_csv("laptop_price.csv",encoding='Latin')

# list of variables for user to input 
list_of_copmanies = df['Company'].value_counts().index
list_of_CPU = df['Cpu'].value_counts().index
list_of_GPU = df['Gpu'].value_counts().index
list_of_Types = df['TypeName'].value_counts().index
list_of_SR = df['ScreenResolution'].value_counts().index
list_of_OpSys = df['OpSys'].value_counts().index
list_of_Memories = df['Memory'].value_counts().index
list_of_Ram = df['Ram'].value_counts().index
list_of_Inches = df['Inches'].value_counts().index

# preprocessing 
def preprocessoring(df):
 
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

    df.Weight = df.Weight.apply(lambda x:(float(x.replace('kg',''))*1000) if 'kg' in x else x )

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
    if 'Product' in df.columns:
        df = df.drop('Product' ,axis = 1 )
        df = df.drop('laptop_ID',axis = 1)
    return df


# editing user input 
def create_input_user(x,user_input_df):
    dict = {}
    for i in x.columns:
        dict[i] = [0] 
    
    user_df = pd.DataFrame(dict)
    user_input = preprocessoring(user_input_df)
    user_input = pd.concat([user_input,user_df])
    user_input = user_input.fillna(0).reset_index().drop(['index','Price_euros'],axis = 1).drop(1,axis = 0)
    return user_input

# SubHeader
slt.subheader('User Input')

# Call preprocessing func 
df = preprocessoring(df)

# show dataframe on website

# sidebar 
slt.sidebar.header('Edit on the Model')

## Feature selection 
nof = slt.sidebar.select_slider("Number of features",[i for i in range(df.shape[1])],15)

# Specific column with higher corr 
feature = abs(df.corr()).sort_values(by = 'Price_euros')[-nof:-1:].index

# Split the data to train and test 
X = df.drop('Price_euros',axis = 1 )
y = df['Price_euros']
X = X[feature]
X_train ,X_test ,y_train , y_test = train_test_split(X,y,test_size=0.25)

# Create dataframe of 1 Row all zeros from the X_train
X_train.iloc[0]=0
user =X_train[X_train.Ram == 0]

# Select Scaler 
Selected = slt.sidebar.selectbox('Select Scaler',['StandardScaler','MinMaxScaler','None'])

if Selected == "StandardScaler":
    Standard_Scaler = 1
    Min_Max_Scaler = 0
elif Selected == "MinMaxScaler":
    Min_Max_Scaler = 1
    Standard_Scaler = 0
else:
    Standard_Scaler =0
    Min_Max_Scaler = 0


# Scale the data with StandardScaler
if Standard_Scaler == 1: 
    SC = StandardScaler()
    SC.fit(X_train)
    X_train = SC.transform(X_train)
    X_test = SC.transform(X_test)

# Scale the data with MinMaxScaler
if Min_Max_Scaler == 1:
    SMM = MinMaxScaler()
    SMM.fit(X_train)
    X_train = SMM.transform(X_train)
    X_test = SMM.transform(X_test)

# Select model to use in machine learing 
select_model = slt.sidebar.selectbox('Select Model', ['Random Forest Regressor','Linear Regression',
                                                      'Neural Networks'])
## Machaine Learing 

def create_model (select_model,X_train,X_test,y_train,y_test):

    # Random Forest Regressor
    if select_model == 'Random Forest Regressor':
        from sklearn.ensemble import RandomForestRegressor
        n_estimatos = slt.sidebar.select_slider("Number of Estimatores ",[i for i in range(1,1001)],100)
        max_features = slt.sidebar.selectbox("select max features ",['sqrt','log2'])
        max_depth = slt.sidebar.select_slider("Number of max depth ",[i for i in range(1,50)],10)

        RFR = RandomForestRegressor(n_estimators=n_estimatos,max_features = max_features ,max_depth = max_depth )
        RFR.fit(X_train, y_train)
        y_hat = RFR.predict(X_test)

        # accuracy check Random Forest Regressor 
        MAE = mean_absolute_error(y_test,y_hat)
        print(MAE)
        MSE = mean_squared_error(y_test,y_hat)
        print(MSE)
        slt.sidebar.text(f'MAE :{MAE}')
        return RFR 

    # Linear Regression
    if select_model == 'Linear Regression':
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_hat = lr.predict(X_test)

        #accuracy check Linear Regression  
        MAE = mean_absolute_error(y_test,y_hat)
        print(MAE)
        MSE = mean_squared_error(y_test,y_hat)
        slt.sidebar.text(f'MAE :{MAE}')
        return lr 
    
    # Tensorflow'
    if select_model =='Neural Networks':
        import tensorflow as tf
        from tensorflow import keras


        # create Sequential model
        model = keras.Sequential()

        number_of_layers = slt.sidebar.select_slider('chosse number of layers',range(1,11))
        for i in range(number_of_layers):
            
            # User input for each layer 
            type_of_layer=slt.sidebar.selectbox(f'Select type of layer {i}',['Dense'])
            type_of_activation=slt.sidebar.selectbox(f'Select type of activation {i}',['relu','sigmoid'])
            number_of_units=slt.sidebar.select_slider(f'chosse number of units {i}',range(1,1000))
            if type_of_layer == 'Dense':
                model.add(tf.keras.layers.Dense(number_of_units,activation=type_of_activation))
            
            if type_of_layer == '2Dcove':
                pass 

        # Compile the model
        model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error              
        optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
        metrics=["mae"])
        
        ## Train the model
        history = model.fit(X_train, y_train, epochs=50, verbose=0)
        

        y_hat = model.predict(X_test)

        #accuracy check Tensorflow model 
        MAE = mean_absolute_error(y_test,y_hat)
        MSE = mean_squared_error(y_test,y_hat)
        slt.sidebar.text(f'MAE :{MAE}')
        
        return model 


# Select User input
Selected_com = slt.selectbox('List of Copmanies',[i for i in list_of_copmanies])
Selected_CPU = slt.selectbox('List of CPUs',[i for i in list_of_CPU])
Selected_GPU = slt.selectbox('List of GPUs',[i for i in list_of_GPU])
Selected_Type = slt.selectbox('List of Types',[i for i in list_of_Types])
Selected_SR = slt.selectbox('List of Screen Resolutions',[i for i in list_of_SR])
Selected_OpSys = slt.selectbox('List of Opreating Systems',[i for i in list_of_OpSys])
Selected_Memory = slt.selectbox('List of Memories',[i for i in list_of_Memories])
Selected_Ram = slt.selectbox('List of Rams',[i for i in list_of_Ram])
Selected_Inche = slt.selectbox('List of Inches',[i for i in list_of_Inches])

# Save selected to dataframe 
user_input_dict ={
       'Company':[Selected_com],
       'Cpu' : [Selected_CPU],
       'Gpu' : [Selected_GPU],
       'TypeName': [Selected_Type],
       'ScreenResolution' : [Selected_SR],
       'OpSys' : [Selected_OpSys],
       'Memory' : [Selected_Memory],
       'Inches' :[Selected_Inche],
       'Ram' : [Selected_Ram],
       'Weight' : [(str(df.Weight.mean()//1000)+'kg')]
       }

model = create_model(select_model, X_train, X_test, y_train, y_test)

user_input_df = pd.DataFrame(user_input_dict)
user_input = create_input_user(df,user_input_df)

for col in user.columns:
    for Ucol in user_input.columns:
        if col == Ucol:
            user[col] =user_input[Ucol][0]

# Scale User Input with StandardScaler
if Standard_Scaler == 1:
    user = SC.transform(user)

# Scale User Input with MinMaxScaler
if Min_Max_Scaler == 1:
    user = SMM.transform(user)
    
price = model.predict(user)
slt.text('')
slt.text(f'Price is : {(np.round(price/5),2)[0][0]}$')
