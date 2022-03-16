from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Создаем модель с архитектурой VGG16 и загружаем веса, обученные
# на наборе данных ImageNet
model = VGG16(weights='imagenet')


def img_to_num(path: str, target_height: int, target_width: int):
    # Загружаем изображение для распознавания, преобразовываем его в массив
    # numpy и выполняем предварительную обработку
    img = image.load_img(path, target_size=(target_height, target_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


arr_path = ['cat', 'ship', 'plane']
for i in range(3):
    path = arr_path[i] + '.jpg'
    picture = img_to_num(path, 224, 224)
    # Запускаем распознавание объекта на изображении
    preds = model.predict(picture)
    # Печатаем три класса объекта с самой высокой вероятностью
    print('Result:', decode_predictions(preds, top=3)[0])
