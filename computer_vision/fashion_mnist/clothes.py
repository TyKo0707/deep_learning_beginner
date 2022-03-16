from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils as utils
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255

y_train = utils.to_categorical(y_train, 10)

classes = ['t-shirt', 'trousers', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

model = Sequential()

model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=200,
                    epochs=5,
                    verbose=1)

model_json = model.to_json()
json_file = open("mnist_model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("mnist_model.h5")

pred = model.predict(x_train)
print(pred[0])
print(f'Predicted class: {classes[np.argmax(pred[0])]}'
      f'Real class: {classes[np.argmax(y_train[0])]}')
