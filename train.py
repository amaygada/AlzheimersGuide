import cv2
import numpy as np
import os 
import time

faces = [] #list to store face pixel matrices
labels =[] #list to store labels
path = './train/' #path where train images are stored
label_str = os.listdir(path) #getting folder names and storing in label_str
print('getting images')
label_count = 1;
for s in label_str:
    path_new = path + str(s) + '/'
    folders = os.listdir(path_new) #getting images from folder
    print('parsing image folder of : ' + str(label_str[label_count-1]))
    for folder in folders: 
        image = cv2.imread(path_new+folder)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) #converting to black and white
        faces.append(image) #appending faces to the list of faces
        labels.append(label_count) #appending labels to the llist of labels
    label_count = label_count + 1 #increasing label count so new label generated for each new folder of images
    
if len(faces)!=0: 

    model = cv2.face.LBPHFaceRecognizer_create()
    t = time.time()
    model.train(np.array(faces),np.array(labels))
    print("time taken: ", time.time() - t)
    print('saving model to disk')
    model.write('lbph_model_trained.xml')
    print('model successfully saved to disk!')

else:
    print('No images found in the folder train')
    print('please run click.py to click images first')
