from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2
import numpy as np

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
batch_size = 16

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)


vgg16_net = VGG16(weights='imagenet',
                  include_top=False,  # Не будет загружено полносвязная часть сети, только свёртка
                  input_shape=(224, 224, 3))

features_train = vgg16_net.predict_generator(train_generator,
                                             nb_train_samples // batch_size
                                             )

features_val = vgg16_net.predict_generator(val_generator,
                                           nb_validation_samples // batch_size
                                           )

features_test = vgg16_net.predict_generator(test_generator,
                                            nb_test_samples // batch_size
                                            )

print(features_train.shape)
print(features_train[0])

print(features_val.shape)
print(features_val[0])

print(features_test.shape)
print(features_test[0])

np.save(open('features_train.npy', 'wb'), features_train)
np.save(open('features_val.npy', 'wb'), features_val)
np.save(open('features_test.npy', 'wb'), features_test)




