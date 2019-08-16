from keras.datasets import imdb 
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
(train_data,train_labels) , (test_data,test_labels) = imdb.load_data(num_words = 10000)

#print(train_data[1])
word_index = imdb.get_word_index()
reversed_word_index = dict([(value,key) for (key,value) in word_index.items()])
#review = ''.join([reversed_word_index.get(i-3,'?') for i in train_data[1]])
#print(review)

def vectorize_seqence(sequences,dimension = 10000):
	results = np.zeros(shape=(len(sequences),dimension))
	print(results.shape)
	for i,sequence in enumerate(sequences):
		#print(i,sequence)
		results[i,sequence] = 1
	return results

#enumerate method will assign a number to each element in the array
# here the the number assigned is the row number of the result matrix which accounts for 
#one review
#The review is basically a set of integers which are mapped to words
#Now the numbers in the review give the location of 1s in the one-hot encoding



x_train = vectorize_seqence(train_data)
x_test = vectorize_seqence(test_data)

y_train = np.asarray(train_labels).astype('float32')
print(y_train.shape)
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1,activation='sigmoid'))


model.compile(optimizer = 'rmsprop',loss = 'mse',metrics = ['accuracy'])
#mse-- mean squared error

# dividing the data into train and validiation data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,batch_size=512,epochs = 20 ,validation_data = (x_val,y_val))
result = model.evaluate(x_test,y_test)
print(result)
