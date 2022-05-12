# Project CV

#### NOTE: make necessary installations using pip install -r requirements.txt in the desired directory

#### (This is written in python 3 So use pip3 instead of pip if working on linux)

#### (If on linux , run using command python3 instead of python which is default for python 2)

### To click training images :

1. run click.py

2. enter your full name

3. look straight into the camera (the program will stop after clicking 10 images)


### To recognize :

1. Run train.py to train the model and store it as an xml file in the disk

2. Run recognize.py on your local machine.

3. The output on the cmd will either be a tupple of (label,distance) or the string "No face found".The string "No face found" indicates no detection. After a significant number of detections (20) the program will end and give result




