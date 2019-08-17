from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
#print(train_data[1])
#print(min(train_targets),max(train_targets))

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(len(train_data[0]),)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer = 'rmsprop' , loss = 'mse',metrics = ['mae'])

#normalisation of data to keep them in same scale

mn = train_data.mean(axis=0)
sd = train_data.std(axis=0)

train_data = train_data-mn
train_data = train_data/sd

test_data -= mn
test_data /= sd

#k-fold validiation

num_samples = len(train_data)//4
all_scores = []
all_mae = []
k = 4

for i in range(k):
	partial_train_data = np.concatenate([train_data[:i*num_samples],train_data[:(i+1)*num_samples]])
	partial_train_targets = np.concatenate([train_targets[:i*num_samples],train_targets[:(i+1)*num_samples]])
	
	val_data = train_data[i*num_samples:(i+1)*num_samples]
	val_targets = train_targets[i*num_samples:(i+1)*num_samples]
	
	history = model.fit(partial_train_data,partial_train_targets,batch_size=1,epochs = 100)
	val_mae,val_mse = model.evaluate(val_data,val_targets)
	#mae_history = history.history['val_mean_absolute_error']
	#all_mae.append(mae_history)
	all_scores.append(val_mae)
	#all_scores.append(history)
	
test_mse,test_mae = model.evaluate(test_data,test_targets)
print(test_mae,test_mse)
print(all_scores)
