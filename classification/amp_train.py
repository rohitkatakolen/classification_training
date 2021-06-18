import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.cuda.amp as amp

class ImageClassificationMixedPrecisionTraining:
	def __init__(self,model, device, EPOCHS, train_loader, val_loader, INTERIM_MODEL_PATH, FINAL_MODEL_PATH):
		self.model = model
		self.device = device
		self.EPOCHS = EPOCHS
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.INTERIM_MODEL_PATH = INTERIM_MODEL_PATH
		self.FINAL_MODEL_PATH = FINAL_MODEL_PATH

	def initiate_training(self):
		self.model = self.model.to(self.device)

		criterion = nn.CrossEntropyLoss()
		#optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.05)


		tr_losses = []
		avg_epoch_tr_loss = []
		tr_accuracy = []

		scaler = amp.GradScaler()

		for epoch in range(self.EPOCHS):
			self.model.train()
			for i, batch in enumerate(self.train_loader):
				data, label = batch
				data, label = data.to(self.device), label.to(self.device)
				optimizer.zero_grad()
				with amp.autocast():
					logit = self.model(data)
					loss = criterion(logit, label)
				scaler.scale(loss).backward()

				scaler.step(optimizer)

				scaler.update()

				tr_losses.append(loss.item())
				tr_accuracy.append(label.eq(logit.argmax(dim=1)).float().mean())
				if i % 50 == 0:
				    print(f'Batch No: {i} \tAverage Training Batch Loss: {torch.tensor(tr_losses).mean():.2f}')
			
			torch.save(self.model, self.INTERIM_MODEL_PATH)

			val_accuracy = []
			self.model.eval()
			for i, batch in enumerate(self.val_loader):
				data, label = batch
				data, label = data.to(self.device), label.to(self.device)
				logit = self.model(data)
				val_accuracy.extend(label.eq(logit.argmax(dim=1)).float().cpu().numpy())

			# Print the training accuracy for each epoch
			print("\n")
			print(f'Epoch No: {epoch + 1}, Training Accuracy: {torch.tensor(tr_accuracy).mean():.2f}')
			
			# Print the validation accuracy for each epoch
			print(f'Epoch No: {epoch + 1}, Validation Accuracy: {torch.tensor(val_accuracy).mean():.2f}')
			print("\n\n")

			# Store the avg epoch loss for plotting
			avg_epoch_tr_loss.append(torch.tensor(tr_losses).mean())

		torch.save(self.model, self.FINAL_MODEL_PATH)

