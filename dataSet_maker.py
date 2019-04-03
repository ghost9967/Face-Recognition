import cv2
import numpy

face_cascade = cv2.CascadeClassifier("C:\\Users\\sarka\\Desktop\\Open CV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
id = input('Enter user id')
photno = 0
if (cam.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cam.isOpened()):
  
  ret, frame = cam.read()
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=5)

    for x,y,w,h in faces:
          photno = photno+1
          cv2.imwrite("dataSet/User."+str(id)+"."+str(photno)+".jpg",gray[y:y+h,x:x+w])
          cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,),3)
          cv2.waitKey(100)

    cv2.imshow('Feed',gray)
    cv2.waitKey(1)
    if(photno>100):break
 
    
cam.release()
cv2.destroyAllWindows()
