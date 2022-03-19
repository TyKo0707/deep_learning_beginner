import dlib
from skimage import io
from scipy.spatial import distance

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


def get_descriptor(img_path: str, ):
    img = io.imread(img_path)
    # win1 = dlib.image_window()
    # win1.clear_overlay()
    # win1.set_image(img)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        print(f'Detection {k}: \n\tLeft: {d.left()} \n\tTop: {d.top()} \n\tRight: {d.right()} \n\tBottom: {d.bottom()}')
        shape = sp(img, d)
        # win1.clear_overlay()
        # win1.add_overlay(d)
        # win1.add_overlay(shape)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        return face_descriptor


face_descriptor1 = get_descriptor('images_to_recognize/sozykin_passport.jpg')
face_descriptor2 = get_descriptor('images_to_recognize/sozykin_webcam.jpg')
s1 = ''
s2 = ''
for i in range(len(face_descriptor1)):
    s1 += '{:>10} '.format(str(round(face_descriptor1[i], 5)))
    s2 += '{:>10} '.format(str(round(face_descriptor2[i], 5)))

print(s1)
print(s2)

euclid_dist = distance.euclidean(face_descriptor1, face_descriptor2)
print(euclid_dist)

