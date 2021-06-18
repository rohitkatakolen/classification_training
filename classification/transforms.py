from torchvision import *

class ImageClassifcationTransforms:
	def __init__(self,TRAIN_DIR, TEST_DIR):
		self.TRAIN_DIR = TRAIN_DIR
		self.TEST_DIR = TEST_DIR

	def get_transforms(self):
		#transforms and data loader
	    mean = [0.485, 0.456, 0.406] 
	    std = [0.229, 0.224, 0.225]

	    train_transforms = transforms.Compose([transforms.Resize((224, 224)), # Resize all images 
	                                       transforms.RandomResizedCrop(224),# Crop
	                                       transforms.RandomRotation(30), # Rotate 
	                                       transforms.RandomHorizontalFlip(), # Flip
	                                       transforms.ToTensor(), # Convert
	                                       transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)) # Normalize
	                                       ])



	    test_transforms = val_transforms = transforms.Compose([transforms.Resize((224, 224)),
	                                         transforms.CenterCrop(224),
	                                         transforms.ToTensor(),
	                                         transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
	                                         ])
	    return train_transforms, test_transforms, val_transforms
