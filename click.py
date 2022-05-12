import cv2
import numpy as np
import os

#function for detecting faces from images
def detect_face_HAAR(image , scaleFactor):
    haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    face = haar_classifier.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=7)
    if len(face)==0:
        return image
    else:
        (x,y,w,h) = face[0]
        dictionary_facedetect = {
            'dim' : face[0],
            'im' : image[y:y+w, x:x+h]
        }
        return dictionary_facedetect
       
       
name = input('please enter your name : ') #this will be the name of the folder where the images of this person will be stored 

path='./train/'
path = os.path.join(path , name) 
os.mkdir(path) #making a directory ./train/<input name>/

count = 0 #setting image count

webcam = cv2.VideoCapture(0) #initializing webcam
while True:
	im_path = path + '/' +str(count) +  '.jpg' #setting name of the image file based on count
	
	try:
		check , frame = webcam.read() #starting webcam
		cv2.imshow("Capturing" , frame) #showing the video of webcam as frames
		#key = cv2.waitKey(1) 
		face = detect_face_HAAR(frame , 1.05) #detecting faces from the frame
		if type(face)==dict:  # if face found
			im = face['im'] #get cropped face image
			cv2.imwrite(filename=im_path, img=im) # save this image
			cv2.waitKey(500) #waiting for 650ms  
			cv2.destroyAllWindows()  
			count = count  + 1; #increasing count
			if count>9: #if 10 images clicked
				print("Images have been succesfully captured")
				print("Turning off camera")
				cv2.destroyAllWindows()
				break; #stop program
				
	except(KeyboardInterrupt):
		print("Turning off camera")
		cv2.destroyAllWindows()
		break;




        
        
        

