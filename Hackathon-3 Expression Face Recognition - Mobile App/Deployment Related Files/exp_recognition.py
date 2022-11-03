import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
# In the below line,remove '.' while working on your local system.However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .exp_recognition_model import *
from PIL import Image
import base64
import io
import os
## Add more imports if required
from torch.autograd import Variable
import torch.nn.functional as F

#############################################################################################################################
#   Caution: Don't change any of the filenames, function names and definitions                                              #
#   Always use the current_path + file_name for refering any files, without it we cannot access files on the server         # 
#############################################################################################################################

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
	

#1) Images captured from mobile is passed as parameter to the below function in the API call, It returns the Expression detected by your network.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Perform necessary transformations to the input(detected face using the above function), this should return the Expression in string form ex: "Anger"
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_expression(img_str):
	imgdata = base64.b64decode(img_str)
	img = Image.open(io.BytesIO(imgdata))
	img =  np.array(img.getdata()).reshape(img.size[1], img.size[0], 3).astype(np.uint8)

        ##########################################################################################
        ##Example: for loading a model use weight state dictionary                              ##
        ##face_det_net = facExpRec()#Example network                                            ##
        ##face_det_net.load_state_dict(torch.load(current_path + '/exp_recognition_net.stdt'))  ##
        ##current_path + '/<network_definition>' is path of the saved model if present in       ##
        ##the same path as this file, we recommend to put in the same directory.                ##
        ##########################################################################################
	
		
	face = detected_face(img)
	if face==0:
		return "No Face Found"

	# YOUR CODE HERE, return expression using your model
	face = trnscm(face)
	myFaceExpModel = facExpRec()
	ckpt = torch.load(current_path + '/face_exp_model_v2.0.t7', map_location=torch.device('cpu'))
	myFaceExpModel.load_state_dict(ckpt['face_exp_dict'])
	prediction = myFaceExpModel(face.unsqueeze(0))
	_, preds = torch.max(prediction, 1)
	exp_Predicted = classes[preds.item()]
	return exp_Predicted
