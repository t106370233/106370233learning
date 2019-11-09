# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:48:23 2019

@author: ahsong
"""
import pandas as pd 
import numpy as np 
from keras.models import Sequential 
from keras import optimizers 
from keras.layers import * 
from keras.callbacks import * 
from sklearn.preprocessing import *
#%%
df_train = pd.read_csv('train-v3.csv') 
df_valid = pd.read_csv('valid-v3.csv') 
df_test = pd.read_csv('test-v3.csv')
#%%
data_title=['bedrooms',
       'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
       'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
       'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
       'sqft_lot15']
df_train.drop(['id'],inplace = True,axis=1) 
df_valid.drop(['id'],inplace = True,axis=1) 
df_test.drop(['id'],inplace = True,axis=1)
Y_train = df_train["price"].values 
X_train = df_train[data_title].values 
Y_valid = df_valid["price"].values 
X_valid = df_valid[data_title].values 
X_test = df_test[data_title].values
#%%
X_train_normal = scale(X_train) 
X_valid_normal = scale(X_valid) 
X_test_normal = scale(X_test)
#%%
model = Sequential() 
model.add(Dense(40, input_dim=X_train.shape[1],  kernel_initializer='normal',activation='relu')) 
model.add(Dense(80, kernel_initializer='normal',activation='relu')) 
model.add(Dense(100, kernel_initializer='normal',activation='relu')) 
model.add(Dense(80, kernel_initializer='normal',activation='relu')) 
model.add(Dense(40, kernel_initializer='normal',activation='relu')) 
model.add(Dense(1,  kernel_initializer='normal'))
model.compile(loss='MAE', optimizer='adam')
#%%
nb_epoch = 500 
batch_size = 32
file_name=str(nb_epoch)+'_'+str(batch_size) 
TB=TensorBoard(log_dir='logs/'+file_name, histogram_freq=0) 
model.fit(X_train_normal, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_valid_normal, Y_valid),callbacks=[TB]) 
#%%
#model.save('h5/'+file_name+'.h5')
Y_predict = model.predict(X_test_normal) 
np.savetxt('test_16.csv', Y_predict, delimiter = ',')
#%%
Y_predict=pd.DataFrame(Y_predict)
Y_predict.to_csv('final7.csv')




#%%

'''
data_2 = pd.read_csv('valid-v3.csv')
X_valid = data_2.drop(['price','id'],axis=1).values
Y_valid = data_2['price'].values

data_3 = pd.read_csv('test-v3.csv')
X_test = data_3.drop(['id'],axis=1).values
#%%
X_train=scale(X_train)
X_valid=scale(X_valid)
X_test=scale(X_test)

#%%


regr = LinearRegression()
regr.fit(X_train, Y_train)
regr.coef_
regr.intercept_
regr.score(X_train,Y_train)
y_test=regr.predict(X_valid)
((Y_valid - regr.predict(X_valid))**2).sum()
final=regr.predict(X_test)
final=pd.DataFrame(final)
#%%
final.to_csv('final3.csv')