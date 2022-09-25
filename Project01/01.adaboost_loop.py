import cv2
import os

datpath = 'data/'
# img = cv2.imread('1.jpg')

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("haarcascade_frontalface_alt2.xml")

for img in os.listdir(datpath):
	frame = cv2.imread(datpath+img)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray,1.3,5)

	for(x,y,w,h) in faces:
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.imwrite('result/'+img,frame)
	# cv2.imshow('img',img)
	# cv2.waitKey()

