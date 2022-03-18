import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

nb_test_samples = 3750

features_test = np.load(open('features_test.npy', 'rb'))

labels_test = np.array([0] * (nb_test_samples // 2) + [1] * (nb_test_samples // 2))

print(labels_test)

model = Sequential()
model.add(Flatten(input_shape=features_test.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(features_test, labels_test,
          epochs=15,
          batch_size=64,
          verbose=2)
