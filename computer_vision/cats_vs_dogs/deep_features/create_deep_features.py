from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2
import numpy as np

directories = ['train', 'val', 'test']

vgg16_net = VGG16(weights='imagenet',
                  include_top=False,  # Не будет загружено полносвязная часть сети, только свёртка
                  input_shape=(224, 224, 3))

nb_samples = {'train': 17500, 'val': 3750, 'test': 3750}

datagen = ImageDataGenerator(rescale=1. / 255)


def create_generator(directory: str, batch_size: int, target_size: tuple):
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    return generator


def create_features(directory: str, batch_size: int, target_size: tuple):
    generator = create_generator(directory, batch_size, target_size)
    features = vgg16_net.predict(generator,
                                 nb_samples[directory] // batch_size
                                 )
    print(features.shape)
    print(features[0])
    np.save(open(f'features_{directory}.npy', 'wb'), features)


batch_size = 16
image_size = (224, 224)
for i in range(len(directories)):
    create_features(directories[i], batch_size, image_size)
