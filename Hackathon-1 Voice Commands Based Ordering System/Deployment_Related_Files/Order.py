#from sklearn.externals import joblib
from joblib import load
from . import Record_audio
import numpy as np
import warnings
import os
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

n_mfcc = 30
n_mels = 128
frames = 15

list_task = [['Idly', ' Dosa', ' Wada', 'Puri ', 'Chapathi'],[0, 1, 2, 3, 4]];
menu_prompts = ["menu_list.wav", "quantity_list.wav"]
#list_task[0] is list of menu items
#list_task[1] is list of quantity allowed

#Caution: DO NOT CHANGE Function Names
####################################################################################
#One way is to regress the features to get the labels
#Get a confidence metric to evaluate your prediction
####################################################################################

T = 0.75 # Confidence threshold

class order():
	#Classify based on the given features and model	
	#It should return predicted label and a confidence measure for your prediction
	def classify_input(self, features, model):
                #YOUR CODE HERE to Reshape the features
                print('test features')
                features1 = np.asarray(features)
                print(features1.shape)
                features = features1.reshape(1, -1)
                print(features.shape)
                
                
                
                #Uncomment the below lines
                
                #YOUR CODE HERE to get the prediction which should be an Integer.
                predicted_label = int(model.predict(features)[0])
                print(predicted_label)
                
                #YOUR CODE HERE to get the confidence measure, Hint: use model.predict_proba
                confidence_measure = np.max(model.predict_proba(features)[0])
                
                return predicted_label, confidence_measure

	def confirm_input(self, digit,confidence,flag):		
		if(confidence > T) and (digit < len(list_task[flag])):
			return digit,list_task[flag][digit]
		return -99, -99


	def take_user_input(self, flag):
                #Extracting features from the user input
                features = Record_audio.get_features(BASE_DIR + "/Hackathon-setup/1_userinput_1.wav", sr=8000)
                print(features.shape)
                model = load(BASE_DIR + '/Hackathon-setup/studio_data_DT.sav') #YOUR CODE HERE to load the model, Hint: use joblib to load the model
                print('model is working')
                #Calling classify_input to get the prediction and confidence measure by giving features and model
                digit,confidence = self.classify_input(features,model)
                digit,choice = self.confirm_input(digit,confidence,flag)
                return digit,choice


