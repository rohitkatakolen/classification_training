import torch

class ImageClassificationTest:
    def __init__(self,model, device, test_loader):
        self.model = model
        self.device = device
        self.test_loader = test_loader

    def initiate_testing(self):
        self.model.eval()
        test_accuracy = []
        for i, batch in enumerate(self.test_loader):
            data, label = batch
            data, label = data.to(self.device), label.to(self.device)
            logit = self.model(data)
            test_accuracy.extend(label.eq(logit.argmax(dim=1)).float().cpu().numpy())

        # Print the test accuracy for each epoch
        print("\n")
        print(f'Test Accuracy: {torch.tensor(test_accuracy).mean():.2f}')
