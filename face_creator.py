import cv2
import numpy as np

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')

cap= cv2.VideoCapture(0)

id= input('enter user id')
sample=0

while 1:
    ret,img= cap.read()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sample+=1
        cv2.imwrite('dataSet/user.'+str(id)+'.'+str(sample)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        cv2.waitKey(100)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]

        eye= eye_cascade.detectMultiScale(roi_gray)

        for(ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

            

    cv2.imshow('sasi',img)
    cv2.waitKey(1)
    if sample>20:
        break

cap.release()
cv2.destroyAllWindows()
    
