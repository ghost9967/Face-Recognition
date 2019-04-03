import cv2
face_cascade = cv2.CascadeClassifier("C:\\Users\\sarka\\Desktop\\Open CV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
axi = cv2.face.LBPHFaceRecognizer_create()
axi.read("axi\\trainingData.yml")
id = 0
font= cv2.FONT_HERSHEY_SIMPLEX
if (cam.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cam.isOpened()):
  
  ret, frame = cam.read()
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors=5)

    for x,y,w,h in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
          id,conf=axi.predict(gray[y:y+h,x:x+w])
          if(id==1):
            id="Arpan"
          cv2.putText(frame,str(id),(x,y+h),font,1,255)

    cv2.imshow('Frame',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    
cam.release()
cv2.destroyAllWindows()
