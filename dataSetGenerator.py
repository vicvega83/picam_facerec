from picamera.array import PiRGBArray
from picamera import PiCamera
import sys
import time
import cv2
#cam = cv2.VideoCapture(0)
#detector=cv2.CascadeClassifier('Classifiers/face.xml')
detector=cv2.CascadeClassifier('/home/pi/haarcascade_frontalface_default.xml')
i=0
offset=50
name=input('enter your id: ')

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.vflip = True
camera.video_stabilization = True
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	
	for(x,y,w,h) in faces:
		i=i+1
		cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
		cv2.rectangle(image,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
		cv2.imshow('frame',image[y-offset:y+h+offset,x-offset:x+w+offset])
		
		cv2.waitKey(100)
	rawCapture.truncate(0)
	if i>20:
		#cam.release()
		cv2.destroyAllWindows()
		break
	
