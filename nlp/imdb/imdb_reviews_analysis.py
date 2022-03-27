from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences

max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

word_index = imdb.get_word_index()

reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key

index = 3
message = ''
for code in x_train[index]:
    word = reverse_word_index.get(code - 3, '?')
    message += word + ' '

maxlen = 200

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=25,
                    batch_size=128,
                    validation_split=0.1)

model.save_weights("C:\\Users\\38097\\Desktop\\deep_learn\\models\\imdb\\imdb.h5")
model_json = model.to_json()
json_file = open("C:\\Users\\38097\\Desktop\\deep_learn\\models\\imdb\\imdb.json", "w")
json_file.write(model_json)
json_file.close()
