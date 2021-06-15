import cv2
import sys
import os
import face_recognition

#dlib CUBA variation
#cascPath = sys.argv[1]    #comment this
#faceCascade = cv2.CascadeClassifier(cascPath) #comment this

video_capture = cv2.VideoCapture(0)

print("processing known faces")

Known_faces_dir = "known_faces_dir"

known_face_encodings = []
known_face_names = []

for name in os.listdir(Known_faces_dir):
    for filename in os.listdir(f"{Known_faces_dir}/{name}"):
        image = face_recognition.load_image_file(f"{Known_faces_dir}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)


'''#Image enter
image = cv2.imread('Kau1.jpg')
Kaushik = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.imread('Papa.jpeg')
Papa = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.imread('Mummy.jpg')
Mummy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Create Face encodings
face_kaushik_encodings = face_recognition.face_encodings(Kaushik)[0]
face_papa_encodings = face_recognition.face_encodings(Papa)[0]
face_mummy_encodings = face_recognition.face_encodings(Mummy)[0]

#Known face database
known_face_encodings = [
    face_kaushik_encodings,
    face_papa_encodings,
    face_mummy_encodings,
]

known_face_names = ["kaushik","Papa",'Mummy']
'''
print("starting camera")
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print('gray')
    face_locations = face_recognition.face_locations(gray,model="hog")
    #face_locations = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    print('location found')
    face_unknown_encodings = face_recognition.face_encodings(frame,face_locations)
    print("here")
    # Draw a rectangle around the faces
    for (top,right,bottom,left) , face_encoding in zip(face_locations , face_unknown_encodings):
        found = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.6)
        name = "unknown"
        print(found)

        if True in found:
            first_match_index = found.index(True)
            name = known_face_names[first_match_index]
            print(name + " found in the image")
            x,y,w,h =  left,top,right,bottom 
            #print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            cv2.rectangle(frame,(x,h),(w,h+22),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x+10,h+12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()