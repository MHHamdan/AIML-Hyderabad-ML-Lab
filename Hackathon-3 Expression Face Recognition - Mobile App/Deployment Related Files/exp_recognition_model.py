import torch
import torchvision
import torch.nn as nn
## Add more imports if required
import torch.nn.functional as F

####################################################################################################################
# Define your model and transform and all necessary helper functions here               		           #
# They will be imported to the exp_recognition.py file    							   # 
####################################################################################################################

# Definition of classes as dictionary
classes = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}

# Example Network
class facExpRec(torch.nn.Module):
	def __init__(self):
		#YOUR CODE HERE
		super(facExpRec, self).__init__()

		# Convolution Layer 1  (1*48 * 48)              (64 * 48 * 48)
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
							  padding=1)  # output size of the first convolutional layer is 64*48*48
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU()
		self.Dropout1 = nn.Dropout(0.2)
		# Maxpool for the Convolutional Layer 1
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		# Maxpooling reduces the size by kernel size. After Maxpooling the output size is 64*24*24

		# YOUR CODE HERE for defining more number of Convolutional layers with Maxpool as required (Hint: Use at least 3 convolutional layers for better performance)
		# Convolution Layer 2 (64 * 24 * 24)              (128 * 24 * 24)
		self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.Dropout2 = nn.Dropout(0.2)
		# Maxpool for the Convolutional Layer 2
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		# Maxpooling reduces the size by kernel size. After Maxpooling the output size is 128 * 12 * 12

		# Convolution Layer 3 (128 *12 * 12)              (512 * 12 * 12)
		self.cnn3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(512)
		self.relu3 = nn.ReLU()
		self.Dropout3 = nn.Dropout(0.2)
		# Maxpool for the Convolutional Layer 3
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		# Maxpooling reduces the size by kernel size. After Maxpooling the output size is 512 * 6 * 6

		# Convolution Layer 4 (512 * 6 * 6)              (512 * 6 * 6)
		self.cnn4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(512)
		self.relu4 = nn.ReLU()
		self.Dropout4 = nn.Dropout(0.2)
		# Maxpool for the Convolutional Layer 4
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		# Maxpooling reduces the size by kernel size. After Maxpooling the output size is 512 * 3 * 3

		# Linear layers
		# (512*3*3) input features, 256 output features
		self.fc1 = nn.Linear(512 * 3 * 3, 256)
		self.fc1 = nn.BatchNorm1d(256)
		self.fc1 = nn.Dropout(0.2)
		self.fc1 = nn.ReLU()
		# 256 input features, 512 output features
		self.fc2 = nn.Linear(256, 512)
		self.fc2 = nn.BatchNorm1d(512)
		self.fc2 = nn.Dropout(0.2)
		self.fc2 = nn.ReLU()
		# 512 input features, 7 output features
		self.fc3 = nn.Linear(512, 7)
		self.fc3 = nn.LogSoftmax(dim=1)

	def forward(self, x):
		# Convolution Layer 1 and Maxpool
		out = self.cnn1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.Dropout1(out)
		out = self.maxpool1(out)

		# YOUR CODE HERE for the Convolutional Layers and Maxpool based on the defined Convolutional layers
		# Convolution Layer 2 and Maxpool
		out = self.cnn2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.Dropout2(out)
		out = self.maxpool2(out)

		# Convolution Layer 3 and Maxpool
		out = self.cnn3(out)
		out = self.bn3(out)
		out = self.relu3(out)
		out = self.Dropout3(out)
		out = self.maxpool3(out)

		# Convolution Layer 4 and Maxpool
		out = self.cnn4(out)
		out = self.bn4(out)
		out = self.relu4(out)
		out = self.Dropout4(out)
		out = self.maxpool4(out)

		# Flattening
		out = out.view(-1, 2048)

		# Linear layers with RELU activation

		# out = self.fc1(out)
		out = self.fc1(out)
		out = self.fc2(out)
		out = self.fc3(out)

		return F.log_softmax(out, dim=1)

	# sample Helper function
def rgb2gray(image):
	return image.convert('L')
	
# Sample Transformation function
#YOUR CODE HERE for changing the Transformation values.
#trnscm = torchvision.transforms.Compose([rgb2gray, torchvision.transforms.Resize((48,48)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trnscm = torchvision.transforms.Compose([rgb2gray, torchvision.transforms.Resize((48,48)), torchvision.transforms.ToTensor()])