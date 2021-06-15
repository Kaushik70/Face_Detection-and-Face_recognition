import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, image = cap.read()
    image = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    
    cv2.imshow('Input', image)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()