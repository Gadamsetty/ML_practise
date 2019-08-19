import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('tata_train.csv')
train_data = data.iloc[:,1:2].values
print(train_data.shape)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_data = sc.fit_transform(train_data)
print(train_data)

x_train = []
y_train = []

for i in range(60,2035):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
print(x_train.shape)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units = 50,return_sequences = True,input_shape =(x_train.shape[1],1) ))
model.add(Dropout(0.2))

model.add(LSTM(units = 50,return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train,y_train,batch_size = 32,epochs = 100)

model.save('lstm_test.h5')

test_data = pd.read_csv('tatatest.csv')
y_test = test_data.iloc[:,1:2].values

total_data = pd.concat((data['Open'],test_data['Open']),axis=0)
inputs = total_data[len(total_data) - len(test_data)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []

for i in range(60,76):
	x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
predicted_stock = model.predict(x_test)
predicted_stock  = sc.inverse_transform(predicted_stock)

plt.plot(y_test,color = 'black')
plt.plot(predicted_stock,color = 'green')
plt.show()
