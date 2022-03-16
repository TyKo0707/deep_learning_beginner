from keras.models import load_model
from keras.preprocessing import image
from IPython.display import Image
import numpy as np

classes = ['t-shirt', 'trousers', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']

model = load_model('fashion_mnist_dense.h5')

print(model.summary())

img_path = 'learn_1.jpg'

Image(img_path, width=150, height=150)

img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

prediction = model.predict(x)
print(prediction)
print(f'Predicted class: {classes[np.argmax(prediction[0])]}\n'
      f'Real class: bag')
