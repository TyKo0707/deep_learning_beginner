import numpy as np
import numpy.random
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

numpy.random.seed(42)

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train -= mean
x_train /= std

x_test -= mean
x_test /= std

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=[x_train.shape[1]]))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=40, validation_split=0.2, verbose=2, batch_size=1)

mse, mae = model.evaluate(x_test, y_test, verbose=0)
print(mae)
