import numpy as np
import torch
from torchvision import *
from torch.utils.data.dataloader import DataLoader

import csv


class ImageClassificationDataLoader:
	num_classes = 0

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
		    train_classes_counts = {}
		    for classes in os.listdir(self.TRAIN_DIR):
		        counts = len(os.listdir(os.path.join(self.TRAIN_DIR, classes)))
		        train_classes_counts[classes] =counts
		    print("Training data: \n", train_classes_counts,'\n')

		if not(os.path.exists(self.TEST_DIR)):
		    print("Test/Validation folder not found")
		else:
		    test_classes_counts = {}
		    for classes in os.listdir(self.TEST_DIR):
		        counts = len(os.listdir(os.path.join(self.TEST_DIR, classes)))
		        test_classes_counts[classes] =counts
		    print("Test data: \n", test_classes_counts,"\n")


		classes = os.listdir(self.TRAIN_DIR)
		classes = {k:v for k , v in enumerate(sorted(classes))}

		self.NUM_CLASSES = len(classes)

		ImageClassificationDataLoader.num_classes = len(classes) # change made

		# change made
		with open('dataset.csv', 'w', newline='') as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames = ['classes', 'train_set', 'test_set'])
			writer.writeheader()

			classes_list = [None]*len(classes)
			train_set_list = [0]*len(classes)
			test_set_list = [0]*len(classes)

			for i in classes:
				classes_list[i] = classes[i] 

			for cls in train_classes_counts:
				train_set_list[classes_list.index(cls)] = train_classes_counts[cls]

			for cls in test_classes_counts:
				test_set_list[classes_list.index(cls)] = test_classes_counts[cls]

			data = {
				'classes' : classes_list,
				'train_set': train_set_list,
				'test_set' : test_set_list
			}
			
			writer.writerow(data)
		# change made


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


