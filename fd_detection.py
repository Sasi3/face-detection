import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

Id=0
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            
        if(Id==1):
            Id="sasi"
        elif(Id==2):
            Id="tarun"
        else:
            Id="Unknown"
            
        cv2.putText(im,str(Id),(x,y+h),font,2,255,2,cv2.LINE_AA)
        
    cv2.imshow('img',im)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
