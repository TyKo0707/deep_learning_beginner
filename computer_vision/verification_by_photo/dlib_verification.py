import dlib
from skimage import io
from scipy.spatial import distance

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

img = io.imread('images_to_recognize/sozykin_passport.jpg')

win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

