from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150

nb_train_samples = 17500
nb_validation_samples = 3750
nb_test_samples = 3750

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 16

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

vgg16_net = VGG16(weights='imagenet',
                  include_top=False,  # Не будет загружено полносвязная часть сети, только свёртка
                  input_shape=(150, 150, 3))

vgg16_net.trainable = False  # Дообучения сети не будет
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block5_conv1' or trainable is True:
        trainable = True
        layer.trainable = True
    print(f'Layer: {layer}\nTrainable:{layer.trainable}')

nb_train_samples = 17500
nb_validation_samples = 3750
nb_test_samples = 3750

datagen = ImageDataGenerator(rescale=1. / 255)
img_width, img_height = 150, 150
batch_size = 16

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

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
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=8,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accuracy: %.2f%%" % (scores[1] * 100))

model.save_weights("cats_vs_dogs.h5")
model_json = model.to_json()
json_file = open("cats_vs_dogs.json", "w")
json_file.write(model_json)
json_file.close()
