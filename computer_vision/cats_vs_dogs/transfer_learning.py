from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2
from c_v_d import train_generator, val_generator, test_generator
from c_v_d import batch_size, nb_test_samples, nb_train_samples, nb_validation_samples
import numpy as np

vgg16_net = VGG16(weights='imagenet',
                  include_top=False,  # Не будет загружено полносвязная часть сети, только свёртка
                  input_shape=(150, 150, 3))

vgg16_net.trainable = False  # Дообучения сети не будет

print(vgg16_net.summary())

model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=adam_v2.Adam(lr=1e-5),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=5,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples//batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save_weights("cats_vs_dogs.h5")
model_json = model.to_json()
json_file = open("cats_vs_dogs.json", "w")
json_file.write(model_json)
json_file.close()
