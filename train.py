import os
import cv2
import numpy
from PIL import Image

axi = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImages_id(path):
	imagePath = [os.path.join(path,f) for f in os.listdir(path)]
	face=[]
	id=[]
	for imagePathr in imagePath:
		faceIm=Image.open(imagePathr).convert('L')
		facepy=numpy.array(faceIm,'uint8')
		idn=int(os.path.split(imagePathr)[-1].split('.')[1])
		face.append(facepy)
		id.append(idn)
		cv2.imshow("Train",facepy)
		cv2.waitKey(10)
	return numpy.array(id), face 

id,face = getImages_id(path)
axi.train(face,id)
axi.write('axi/trainingData.yml')
cv2.destroyAllWindows()
