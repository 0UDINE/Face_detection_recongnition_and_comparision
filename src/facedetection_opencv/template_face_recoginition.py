import sys

import cv2
import os
from sklearn.metrics import mean_squared_error

template_path  = '../../images/template.jpg'
folder_path = '../../images/'

# standarizer la taille du visage
img = cv2.imread(template_path)
gray_template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_ref = face_cascade.detectMultiScale(gray_template, scaleFactor=1.1, minNeighbors=5)

if len(face_ref) > 0:
    (x,y,w,h) = face_ref[0]
    # Normalisation
    roi_ref = cv2.resize(gray_template[y:y+h, x:x+w], (200, 200)) # Taille standard


def face_size_normalisation(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if(len(faces) > 0):
        (x,y,w,h) = faces[0]
        roi_face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))# Même taille que la référence !
        return roi_face
    return None


# partie de comparaison entre template et les autres images
if __name__ == '__main__':
    for file_name in os.listdir('../../images/'):
        if file_name.endswith('.jpg') and file_name != 'template.jpg':
            full_path = os.path.join(folder_path, file_name)
            current_face = face_size_normalisation(full_path)
            mse = mean_squared_error(roi_ref.flatten(), current_face.flatten())
            print(f"Image: {file_name} | MSE: {mse:.2f}")

            if mse < 3000:  # Seuil de similarité
                print(f"==> MATCH TROUVÉ : {file_name}")
            print(mse)


