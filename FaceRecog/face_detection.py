import cv2
import sys
import face_recognition

video_capture = cv2.VideoCapture(0)

#Image enter


while True:
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")
    # Capture frame-by-frame
    ret, image = video_capture.read()
    image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    #image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image, model='hog')

    number_of_faces = len(face_locations)
    print(number_of_faces)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        x, y, w, h = left, top, right, bottom

        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
