import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_camera = cv2.VideoCapture(0)

while 1:
    _, img = my_camera.read()

    # convert to gray scale of each frames
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]


    cv2.imshow('img', img)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

# Close the window
my_camera.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
