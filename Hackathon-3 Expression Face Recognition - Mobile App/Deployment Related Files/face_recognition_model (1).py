import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
## Add more imports if required

##Sample Transformation function
#YOUR CODE HERE for changing the Transformation values.
#trnscm = torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Added code below by removing the normalizing as it was not done in the collab notebook hence keeping it the same.
trnscm = torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)), torchvision.transforms.ToTensor()])
trnscm_custom = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])
#trnscm_mlp_recog = torchvision.transforms.Compose([torchvision.transforms.Resize((100,100)), torchvision.transforms.ToTensor()])
#trnscm_custom_recog = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])

##Example Network
class SiameseNetwork(torch.nn.Module):
	def __init__(self):
		#YOUR CODE HERE - Added code below
		super(SiameseNetwork, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.ReflectionPad2d(1),
			# Pads the input tensor using the reflection of the input boundary, it similar to the padding.
			nn.Conv2d(1, 4, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(4),

			nn.ReflectionPad2d(1),
			nn.Conv2d(4, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),

			nn.ReflectionPad2d(1),
			nn.Conv2d(8, 8, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(8),
		)

		self.fc1 = nn.Sequential(
			nn.Linear(8 * 100 * 100, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 500),
			nn.ReLU(inplace=True),

			nn.Linear(500, 5))

	# YOUR CODE HERE - Added code below

	def forward_once(self, x):
		output = self.cnn1(x)
		output = output.view(output.size()[0], -1)
		output = self.fc1(output)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1, output2
		
##Sample classifier Network 
#classifier = nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 16))
classifier = nn.Sequential(nn.Linear(100, 100), nn.InstanceNorm1d(100), nn.ReLU(), nn.Linear(100, 32), nn.InstanceNorm1d(32), nn.ReLU(), nn.Linear(32, 5))

##Definition of classes as dictionary
#classes = ['person1','person2','person3','person4','person5','person6','person7']
#classes = ['person1','person2','person3','person4','person5']
classes = ['Subrot','Sabitha','Sreenath','Harita','Ravi']

# Custom Network
class CustomNetwork(nn.Module):
	def __init__(self):
		super(CustomNetwork, self).__init__()
		# 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(1, 32, 5)

		## Note that among the layers to add, consider including:
		# maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.pool3 = nn.MaxPool2d(2, 2)
		self.conv4 = nn.Conv2d(128, 256, 3)
		self.pool4 = nn.MaxPool2d(2, 2)
		self.conv5 = nn.Conv2d(256, 512, 1)
		self.pool5 = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(512 * 6 * 6, 1024)
		self.fc2 = nn.Linear(1024, 136)

		self.drop1 = nn.Dropout(p=0.1)
		self.drop2 = nn.Dropout(p=0.2)
		self.drop3 = nn.Dropout(p=0.25)
		self.drop4 = nn.Dropout(p=0.25)
		self.drop5 = nn.Dropout(p=0.3)
		self.drop6 = nn.Dropout(p=0.4)

	def forward(self, x):
		## TODO: Define the feedforward behavior of this model
		## x is the input image and, as an example, here you may choose to include a pool/conv step:
		## x = self.pool(F.relu(self.conv1(x)))

		x = self.pool1(F.relu(self.conv1(x)))
		x = self.drop1(x)
		x = self.pool2(F.relu(self.conv2(x)))
		x = self.drop2(x)
		x = self.pool3(F.relu(self.conv3(x)))
		x = self.drop3(x)
		x = self.pool4(F.relu(self.conv4(x)))
		x = self.drop4(x)
		x = self.pool5(F.relu(self.conv5(x)))
		x = self.drop5(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.drop6(x)
		x = self.fc2(x)
		# a modified x, having gone through all the layers of your model, should be returned
		return x