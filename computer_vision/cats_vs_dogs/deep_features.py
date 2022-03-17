from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2

nb_train_samples = 17500
batch_size = 10

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    'train',
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

print(features_train.shape)

