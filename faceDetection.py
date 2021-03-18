import cv2

face_cas = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eye_cas = cv2.CascadeClassifier("haarcascade-eye.xml")

def detect(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 3, 8)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cas.detectMultiScale(roi_gray, 1.8,4)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+ eh), (255, 0, 0), 2)
    return frame

cap = cv2.VideoCapture(0)
while 1:
    ret,frame = cap.read()
    canvas = detect(frame)
    cv2.imshow("video", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"): # q key is exit
        break
cap.release()
cv2.destroyAllWindows()