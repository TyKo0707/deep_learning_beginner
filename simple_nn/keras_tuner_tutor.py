from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils as utils
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Dense(units=hp.Int('units_input',  # Полносвязный слой с разным количеством нейронов
                                 min_value=512,  # минимальное количество нейронов - 128
                                 max_value=1024,  # максимальное количество - 1024
                                 step=32),
                    input_dim=784,
                    activation=activation_choice))
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=hp.Int('units_hidden',
                                    min_value=128,
                                    max_value=600,
                                    step=32),
                        activation=activation_choice))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,                # функция создания модели
    objective='val_accuracy',   # метрика, которую нужно оптимизировать -
                                # доля правильных ответов на проверочном наборе данных
    max_trials=10,              # максимальное количество запусков обучения
    directory='test_directory'  # каталог, куда сохраняются обученные сети
)

print(tuner.search_space_summary())

tuner.search(x_train,                  # Данные для обучения
             y_train,                  # Правильные ответы
             batch_size=100,           # Размер мини-выборки
             epochs=100,                # Количество эпох обучения
             validation_split=0.2,     # Часть данных, которая будет использоваться для проверки
             verbose=1,
             )

# print(tuner.results_summary())
#
# models = tuner.get_best_models(num_models=3)
#
# for model in models:
#     model.summary()
#     model.evaluate(x_test, y_test)
