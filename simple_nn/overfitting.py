from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils as utils
from keras.models import load_model
from keras.models import model_from_json
import h5py
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

classes = ['t-shirt', 'trousers', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

# model = Sequential()
#
# model.add(Dense(800, input_dim=784, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
#
# print(model.summary())
#
# history = model.fit(x_train, y_train,
#                     batch_size=200,
#                     epochs=5,
#                     validation_split=0.2,
#                     verbose=1)
#
# model.save('fashion_mnist_dense.h5')

# scores = model.evaluate(x_test, y_test, verbose=1)
# print(f'Success percent in test dataset: {round(scores[1] * 100, 4)}')

filename = "fashion_mnist_dense.h5"
model = load_model(filename)
pred = model.predict(x_test)
print(pred[0], classes[np.argmax(pred[0])])

json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mnist_model.h5")
