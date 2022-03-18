from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2

# Размеры изображения
target_size = (150, 150)
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (target_size[0], target_size[1], 3)
batch_size = 64
nb_samples = {'train': 17500, 'val': 3750, 'test': 3750}
directories = ['small_dataset/train', 'small_dataset/val', 'small_dataset/test']


def create_datagen(directory: str, size: tuple, batch: int):
    if directory == directories[0]:
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(directory,
                                            target_size=size,
                                            batch_size=batch,
                                            class_mode='binary')

    return generator


vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg16_net.trainable = False  # Дообучения сети не будет
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block5_conv1' or trainable is True:
        trainable = True
        layer.trainable = True
    print(f'Layer: {layer}\nTrainable:{layer.trainable}')
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоя
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=adam_v2.Adam(lr=1e-5),
              metrics=['accuracy'])

model.fit_generator(
    create_datagen(directories[0], target_size, batch_size),
    steps_per_epoch=nb_samples['train'] // batch_size,
    epochs=2,
    validation_data=create_datagen(directories[1], target_size, batch_size),
    validation_steps=nb_samples['val'] // batch_size)

scores = model.evaluate_generator(create_datagen(directories[2], target_size, batch_size), nb_samples['test'] // batch_size)
print("Accuracy: %.2f%%" % (scores[1] * 100))
