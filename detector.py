from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model=load_model('facefeatures_model2.h5')

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture=cv2.VideoCapture(0)
while True:
	ret,frame=capture.read()
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	for(x,y,w,h) in faces:
		ROI=frame[y:y+h,x:x+w]
		for f in faces:
			new_array=cv2.resize(ROI,(224,224))
			im=Image.fromarray(new_array,'RGB')
			img_array=np.array(im)
			img_array=np.expand_dims(img_array,axis=0)

			pred=model.predict(img_array)
			print(pred)

			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			if pred[0][0]>0.9:
				cv2.putText(frame,'Abhinav',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][1]>0.9:
				cv2.putText(frame,'Anukriti',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][2]>0.9:
				cv2.putText(frame,'Geeta',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)	
			elif pred[0][3]>0.9:
				cv2.putText(frame,'Kanish',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][4]>0.9:
				cv2.putText(frame,'Lalit',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][5]>0.9:
				cv2.putText(frame,'Prashant',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][6]>0.9:
				cv2.putText(frame,'Sushil',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][7]>0.9:
				cv2.putText(frame,'Swati',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			elif pred[0][8]>0.9:
				cv2.putText(frame,'Vandana',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
	cv2.imshow("Frame",frame)
	if cv2.waitKey(1) & 0xFF==ord("q"):
		break
capture.release()
cv2.destroyAllWindows()

# def face_extractor(img):
# 	faces=face_cascade.detectMultiScale(img,1.3,5)

# 	if faces is ():
# 		return None
	
# 	for (x,y,w,h) in faces:
# 		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
# 		cropped_face=img[y:y+h,x:x+w]

# 	return cropped_face

# video_capture=cv2.VideoCapture(0)
# while True:
# 	_, frame=video_capture.read()

# 	face=face_extractor(frame)
# 	if type(face) is np.ndarray:
# 		face=cv2.resize(face,(224,224))
# 		im=Image.fromarray(face,'RGB')
# 		img_array=np.array(im)
# 		img_array=np.expand_dims(img_array,axis=0)
# 		pred=model.predict(img_array)
# 		print(pred)

# 		name="None Matching"

# 		if pred[0][0]>0.5:
# 			name='Abhinav'
# 		elif pred[0][1]>0.5:
# 			name='Geeta'
# 		elif pred[0][2]>0.5:
# 			name='Swati'

# 		cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
# 	else:
# 		cv2.putText(frame,"No face found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

# 	cv2.imshow('Video',frame)
# 	if cv2.Waitkey(1) & 0xFF==ord('q'):
# 		break
# video_capture.release()
# cv2.destroyAllWindows()