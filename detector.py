import cv2,os
from picamera.array import PiRGBArray
from picamera import PiCamera
import sys
import time

import numpy as np
from PIL import Image 
import pickle

recsize=30

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.vflip = True
camera.video_stabilization = True
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

#cam = cv2.VideoCapture(0)
#font = cv2.initFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) #Creates a font
font=cv2.FONT_HERSHEY_DUPLEX
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	for(x,y,w,h) in faces:
		nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
		cv2.rectangle(image,(x-recsize,y-recsize),(x+w+recsize,y+h+recsize),(225,0,0),2)
		if(nbr_predicted==2):
			nbr_predicted='Test'
		elif(nbr_predicted==1):
			nbr_predicted='Vik'
		cv2.putText(image, str(nbr_predicted), (x,y-40), font, 0.5, (255,255,255))
	cv2.imshow('frame',image)		
	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
	if key == ord("q"):
		print ("shutting down")
		break








