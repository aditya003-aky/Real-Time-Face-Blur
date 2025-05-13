import cv2
capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    success, img = capture.read()
    
    if not success:
        print("Error: Failed to capture image")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if len(faces) == 0:
        cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    else:
        for (x, y, w, h) in faces:
            face_region = img[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face_region, (91, 91), 0)
            img[y:y + h, x:x + w] = blurred_face

    cv2.imshow('Face Blur', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
