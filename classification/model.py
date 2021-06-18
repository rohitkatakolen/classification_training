import torch
import torchvision
from torchvision import models
import torch.nn as nn
from collections import OrderedDict

import os

class ImageClassificationModel:
	ALEXNET_MODEL_FAMILY = ['alexnet']
	SQUEEZENET_MODEL_FAMILY = ['squeezenet1_0','squeezenet1_1']
	RESNET_MODEL_FAMILY = ['resnet18','resnet34','resnet50','resnet101','resnet152']
	DENSENET_MODEL_FAMILY = ['densenet121','densenet161','densenet169','densenet201']
	VGGNET_MODEL_FAMILY = ['vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn']
	MOBILENET_MODEL_FAMILY = ['mobilenet_v2']

	def __init__(self, NUM_CLASSES, CLASSIFICATION_NUM_HIDDEN_LAYERS, CLASSIFIER_LAYER_SIZE, CLASSIFIER_LAYER_DROPOUT_PROB, MODEL_ARCH):
		self.classifer_num_hidden_layers = CLASSIFICATION_NUM_HIDDEN_LAYERS
		self.classifier_layer_size = CLASSIFIER_LAYER_SIZE
		self.num_classes = NUM_CLASSES
		self.classifier_layer_dropout_prob = CLASSIFIER_LAYER_DROPOUT_PROB
		self.model_arch = MODEL_ARCH

	def get_alexnet_model(self):
		if self.model_arch == 'alexnet':
			model = models.alexnet(pretrained=True)

		for param in model.parameters():
			param.requires_grad = False

		#AlexNet has classifier modules at the end which has 6 layers.
		# Pull output feature dimensions from the final layer from classifier module 
		out_features =  model.classifier[-1].out_features

		#Add final layer to classifier module of alexnet
		model.classifier.add_module('relu1', nn.ReLU())
		model.classifier.add_module('dropout1', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc1', nn.Linear(out_features, self.classifier_layer_size))
		model.classifier.add_module('relu2', nn.ReLU())
		model.classifier.add_module('dropout2', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc2', nn.Linear(self.classifier_layer_size, self.num_classes))

		return model

	def get_densenet_model(self):
		if self.model_arch == 'densenet121':
			model = models.densenet121(pretrained=True)
		elif self.model_arch == 'densenet161':
			model = models.densenet161(pretrained=True)
		elif self.model_arch == 'densenet169':
			model = models.densenet169(pretrained=True)
		elif self.model_arch == 'densenet201':
			model = models.densenet201(pretrained=True)
		else:
			print("Model selection error")
			return None

		for param in model.parameters():
			param.required_grad = False

		#DenseNet classifier module has 1 layer
		# Pull input and  output feature dimensions of this the layer
		in_features, out_features =  model.classifier.in_features, model.classifier.out_features

		#Create a new classifier with first layer same as original and add custom layer below it
		classifier_new = []
		#Keep the first layer same as the original resnet model
		classifier_new.append(('fc1', nn.Linear(in_features, out_features)))
		classifier_new.append(('relu1', nn.ReLU()))
		classifier_new.append(('dropout1', nn.Dropout(self.classifier_layer_dropout_prob)))
		#Adding new layer based on num of classes
		classifier_new.append(('fc2', nn.Linear(out_features, self.num_classes)))
		classifier_new = nn.Sequential(OrderedDict(classifier_new))

		model.classifier = classifier_new

		return model

	def get_vggnet_model(self):
		if self.model_arch == 'vgg11':
			model = models.vgg11(pretrained=True)
		elif self.model_arch == 'vgg13':
			model = models.vgg13(pretrained=True)
		elif self.model_arch == 'vgg16':
			model = models.vgg16(pretrained=True)
		elif self.model_arch == 'vgg19':
			model = models.vgg19(pretrained=True)
		elif self.model_arch == 'vgg11_bn':
			model = models.vgg11_bn(pretrained=True)
		elif self.model_arch == 'vgg13_bn':
			model = models.vgg13_bn(pretrained=True)
		elif self.model_arch == 'vgg16_bn':
			model = models.vgg16_bn(pretrained=True)
		elif self.model_arch == 'vgg19_bn':
			model = models.vgg19_bn(pretrained=True)
		else:
			print("Model selection error")
			return None

		for param in model.parameters():
			param.required_grad = False

		#AlexNet has classifier modules at the end which has 6 layers.
		# Pull output feature dimensions from the final layer from classifier module 
		out_features =  model.classifier[-1].out_features

		#Add final layer to classifier module of alexnet
		model.classifier.add_module('relu1', nn.ReLU())
		model.classifier.add_module('dropout1', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc1', nn.Linear(out_features, self.classifier_layer_size))
		model.classifier.add_module('relu2', nn.ReLU())
		model.classifier.add_module('dropout2', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc2', nn.Linear(self.classifier_layer_size, self.num_classes))

		return model


	def get_resnet_model(self):
		if self.model_arch == 'resnet18':
			model = models.resnet18(pretrained=True)
		elif self.model_arch == 'resnet34':
			model = models.resnet34(pretrained=True)
		elif self.model_arch == 'resnet50':
			model = models.resnet50(pretrained=True)
		elif self.model_arch == 'resnet101':
			model = models.resnet101(pretrained=True)
		elif self.model_arch == 'resnet152':
			model = models.resnet152(pretrained=True)
		else:
			print("Model selection error")
			return None
		
		train_last_layer_only = False
		if(train_last_layer_only):
			print("training only last layer \n")
			for param in model.parameters():
				param.required_grad = False
		else:
			print("Training all layers \n")

		# Pull final fc layer feature dimensions
		in_features, out_features = model.fc.in_features, model.fc.out_features

		# Resnet models have 'fc' as final layer with one linear layer
		# Keep the linear layer as is and add any other layers below it
		# Create custom classifier according based on num of classes
		# Create new classifier and replace with old 'fc' layer for resnet models

		fc_new = []
		#Keep the first layer same as the original resnet model
		fc_new.append(('fc1', nn.Linear(in_features, out_features)))
		fc_new.append(('relu1', nn.ReLU()))
		fc_new.append(('dropout1', nn.Dropout(self.classifier_layer_dropout_prob)))
		#Adding new layer based on num of classes
		fc_new.append(('fc2', nn.Linear(out_features, self.num_classes)))
		fc_new = nn.Sequential(OrderedDict(fc_new))

		model.fc = fc_new

		return model

	def get_mobilenet_model(self):
		if self.model_arch == 'mobilenet_v2':
			model = models.mobilenet_v2(pretrained=True)

		for param in model.parameters():
			param.required_grad = False

		#MobileNet has classifier modules at the end which has 2 layers.
		# Pull output feature dimensions from the final layer from classifier module 
		out_features =  model.classifier[-1].out_features

		#Add final layer to classifier module of alexnet
		model.classifier.add_module('relu1', nn.ReLU())
		model.classifier.add_module('dropout1', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc1', nn.Linear(out_features, self.classifier_layer_size))
		model.classifier.add_module('relu2', nn.ReLU())
		model.classifier.add_module('dropout2', nn.Dropout(self.classifier_layer_dropout_prob))
		model.classifier.add_module('fc2', nn.Linear(self.classifier_layer_size, self.num_classes))

		return model

	def get_squeezenet_model(self):
		if self.model_arch == 'squeezenet1_0':
			model = models.squeezenet1_0(pretrained=True)
		elif self.model_arch == 'squeezenet1_1':
			model = models.squeezenet1_1(pretrained=True)
		else:
			print("Model selection error")
			return None

		# for param in model.parameters():
		# 	param.required_grad = False

		#MobileNet has classifier modules at the end which has 2 layers.
		model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))

		return model

	def initialise_model(self,INTERIM_MODEL_PATH):

		# If it belongs to AlexNet Model architecture
		if self.model_arch in ImageClassificationModel.ALEXNET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_alexnet_model()

		# If it belongs to DenseNet Model architecture
		elif self.model_arch in ImageClassificationModel.DENSENET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_densenet_model()

		# If it belongs to VggNet Model architecture
		elif self.model_arch in ImageClassificationModel.VGGNET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_vggnet_model()

		# If it belongs to Resnet Model architecture
		elif self.model_arch in ImageClassificationModel.RESNET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_resnet_model()

		# If it belongs to MobileNet Model architecture
		elif self.model_arch in ImageClassificationModel.MOBILENET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_mobilenet_model()

		# If it belongs to Squeezenet Model architecture
		elif self.model_arch in ImageClassificationModel.SQUEEZENET_MODEL_FAMILY:
			print("Model architecture :",self.model_arch)
			model = self.get_squeezenet_model()

		# Error handling
		else:
			print("No such pre-trained model")
			return None

		if(os.path.exists(INTERIM_MODEL_PATH)):
			print("Using last saved model \n")
			model = torch.load(INTERIM_MODEL_PATH)

		else:
			print("Creating new model \n")

		return model