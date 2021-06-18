import numpy as np
import torch
from torchvision import *
from torch.utils.data.dataloader import DataLoader



class ImageClassificationDataLoader:
	def __init__(self, TRAIN_DIR, TEST_DIR, BATCH_SIZE, train_transforms, val_transforms, test_transforms, TRAIN_VAL_SPLIT):
		self.TRAIN_DIR = TRAIN_DIR
		self.TEST_DIR = TEST_DIR
		self.BATCH_SIZE = BATCH_SIZE
		self.train_transforms = train_transforms
		self.val_transforms = val_transforms
		self.test_transforms = test_transforms
		self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
		self.NUM_CLASSES = 0


	def get_data_loader(self):
		if not(os.path.exists(self.TRAIN_DIR)):
			print("Train folder not found")
			return
		else:
		    classes_counts = {}
		    for classes in os.listdir(self.TRAIN_DIR):
		        counts = len(os.listdir(os.path.join(self.TRAIN_DIR, classes)))
		        classes_counts[classes] =counts
		    print("Training data: \n",classes_counts,'\n')

		if not(os.path.exists(self.TEST_DIR)):
		    print("Test/Validation folder not found")
		else:
		    classes_counts = {}
		    for classes in os.listdir(self.TEST_DIR):
		        counts = len(os.listdir(os.path.join(self.TEST_DIR, classes)))
		        classes_counts[classes] =counts
		    print("Test data: \n",classes_counts,"\n")

		classes = os.listdir(self.TRAIN_DIR)
		classes = {k:v for k , v in enumerate(sorted(classes))}

		self.NUM_CLASSES = len(classes)

		train_data = datasets.ImageFolder(self.TRAIN_DIR, transform=self.train_transforms)
		test_data = datasets.ImageFolder(self.TEST_DIR, transform=self.test_transforms)

		dataset_size = len(train_data)
		split = int(np.floor(self.TRAIN_VAL_SPLIT * dataset_size))
		train_size, val_size = split, dataset_size - split
		print("Train size: ",train_size)
		print("Validation size: ",val_size)
		print("\n")

		train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.BATCH_SIZE)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.BATCH_SIZE)


		
		return train_loader, val_loader, test_loader

	def get_num_classes(self):
		return self.NUM_CLASSES


