import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
# In the below line,remove '.' while working on your local system. However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .face_recognition_model import *
from PIL import Image, ImageOps
import base64
import io
import os
import joblib
import pickle
## Add more imports if required
# Added the below code
from torch.autograd import Variable
import torch.nn.functional as F


###########################################################################################################################################
#		 Caution: Don't change any of the filenames, function names and definitions                                              #
#		Always use the current_path + file_name for refering any files, without it we cannot access files on the server          # 
##########################################################################################################################################

# Current_path stores absolute path of the file from where it runs. 
current_path = os.path.dirname(os.path.abspath(__file__))

#1) The below function is used to detect faces in the given image.
#2) It returns only one image which has maximum area out of all the detected faces in the photo.
#3) If no face is detected,then it returns zero(0).

def detected_face(image):
	eye_haar = current_path + '/haarcascade_eye.xml'
	face_haar = current_path + '/haarcascade_frontalface_default.xml'
	face_cascade = cv2.CascadeClassifier(face_haar)
	eye_cascade = cv2.CascadeClassifier(eye_haar)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	face_areas=[]
	images = []
	required_image=0
	for i, (x,y,w,h) in enumerate(faces):
		face_cropped = gray[y:y+h, x:x+w]
		face_areas.append(w*h)
		images.append(face_cropped)
		required_image = images[np.argmax(face_areas)]
		required_image = Image.fromarray(required_image)
	return required_image


#1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
#Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(image1, image2):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	imgdata1 = base64.b64decode(image1)
	img1 = Image.open(io.BytesIO(imgdata1))
	img1 =  np.array(img1.getdata()).reshape(img1.size[1], img1.size[0], 3).astype(np.uint8)
	imgdata2 = base64.b64decode(image2)
	img2 = Image.open(io.BytesIO(imgdata2))
	img2 =  np.array(img2.getdata()).reshape(img2.size[1], img2.size[0], 3).astype(np.uint8)
	det_img1 = detected_face(img1)
	det_img2 = detected_face(img2)
	if(det_img1 == 0 or det_img2 == 0):
		return "Face not found"
	face1 = trnscm(det_img1)
	face2 = trnscm(det_img2)
	##########################################################################################
        ##Example for loading a model using weight state dictionary:                            ##
        ##feature_net = light_cnn()#Example network                                             ##
        ##feature_net.load_state_dict(torch.load(current_path + '/siamese_model.t7'))           ##
        ##current_path + '/<network_definition>' is path of the saved model if present in       ##
        ##the same path as this file, we recommend to put in the same directory                 ##
        ##########################################################################################
	##########################################################################################
	
	
	#YOUR CODE HERE, return similarity measure using your model
	# Added the below code
	feature_net = SiameseNetwork()
	ckpt = torch.load(current_path + '/siamese_model.t7',map_location=torch.device('cpu'))
	feature_net.load_state_dict(ckpt['net_dict'])
	output1, output2 = feature_net(Variable(face1.unsqueeze(0)), Variable(face2.unsqueeze(0)))
	euclidean_distance = F.pairwise_distance(output1, output2)
	return euclidean_distance.item()
	
#1) Image captured from mobile is passed as parameter to this function in the API call, It returns the face class in the string form ex: "Person1"
#2) The image is passed to the function in base64 encoding, Code to decode the image provided within the function
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
#4) Perform necessary transformations to the input(detected face using the above function).
#5) Along with the siamese, you need the classifier as well, which is to be finetuned with the faces that you are training
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_face_class(image1):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	imgdata1 = base64.b64decode(image1)
	img1 = Image.open(io.BytesIO(imgdata1))
	img1 =  np.array(img1.getdata()).reshape(img1.size[1], img1.size[0], 3).astype(np.uint8)
	det_img1 = detected_face(img1)
	if(det_img1 == 0):
		return "Face not found"
	##YOUR CODE HERE, return face class here
	##Hint: you need a classifier finetuned for your classes, it takes o/p of siamese as i/p to it
	##Better Hint: Siamese experiment is covered in one of the labs

	face = trnscm_custom(det_img1)
	myCustomModel = CustomNetwork()
	ckpt = torch.load(current_path + '/custom_model_V4.0.t7', map_location=torch.device('cpu'))
	myCustomModel.load_state_dict(ckpt['custom_dict'])
	prediction = myCustomModel(face.unsqueeze(0))
	_, preds = torch.max(prediction, 1)
	#person_Predicted = classes[torch.argmax(preds)]
	person_Predicted = classes[preds.item()]

	return person_Predicted

                                                                                                                                                                                

