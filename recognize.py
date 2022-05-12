import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
import requests
import time

#function to detect faces using haar
def detect_face_HAAR(image , scaleFactor):
    haar_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #making haar cascade object
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) #converting image to black and white
    face = haar_classifier.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=7) #getting result of haar classifier
    if len(face)==0: #checking if haar detected a face
        return image
    else: #else storing dimensions and cropped image in a dictionary
        (x,y,w,h) = face[0] #dimensions
        dictionary_facedetect = {
            'dim' : face[0],
            'im' : image[y:y+w, x:x+h] #cropped image
        }
        return dictionary_facedetect
    
#function to make prediction on the face    
def predict(image):
    test = image.copy() #making copy of the image
    face = detect_face_HAAR(test , 1.05) #detecting the face from the image
    if type(face)==dict: #checking if face detectd
        recog = True 
        (x,y,w,h) = face['dim'] #extracting dimensions
        im = face['im'] #extracting face
        result = model.predict(im) #storing result of prediction in model
        cv2.rectangle(test, (x,y), (x+w, y+h), (0,255,0), 2) #making rectangle around detected face
        print(result) 
        if result[1]<80: #setting a threshold for the obtained eucladian distance
            predictionList.append(result[0]) #appending the label to the prectionList
        return(test)
    else: #if face not detected
        print('No face found!') 
        return (test)


predictionList = [] #list to store prediction results

#to check if faces.py has been run
try:
    path = './train/'
    label_str = os.listdir(path) #np.load('lab_str.npy',allow_pickle=True)
    print('reading model')
    model = cv2.face.LBPHFaceRecognizer_create() #creating the model for LBPH
    model.read('lbph_model_trained.xml')
    print('model ready!')
    
    print('loading the model')
    count = 1
    webcam = cv2.VideoCapture(0) #initiallizing webcam
    while True:
        try:
            t = time.time()
            
            check, frame = webcam.read() #starting  webcam
            frame = predict(frame) #predicting the frame of th desktop
            cv2.imshow("Capturing", frame) #showing the video of webcam frame by frame
            key = cv2.waitKey(20)
            print("count :", count);
            count += 1
            print("time taken:", time.time()-t)
            if len(predictionList)>20: #checking if 20 predictions are made 
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                modeVal = stats.mode(predictionList)[0][0] #taking mode of the labels from predictionList
                recog_str = label_str[modeVal-1] #getting name of the recognized person
                print('Hello ',recog_str) 
                
                try:
                    parameters = {"id":"face_recog_sheet", "Name":str(recog_str)} 
                    URL = "https://script.google.com/macros/s/AKfycbw228iXZD24ubizFjcBx5LZReyQKMMFpVuFSJLKKWypdcracPnu/exec"
                    response = requests.get(URL, params=parameters)
                    print(response.content)
                except:
                    print('Error in sending data to spreadsheet')
                cv2.destroyAllWindows()
                break
    
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
except:
    print('make sure you have run train.py to train the model before running this file')
    print('If you have run it then check if all the packages/modules are correctly installed')
    
