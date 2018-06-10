import numpy as np

import torch
import torchvision


from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Data loading and transforming

# Read image as a tensor
data_transform = transforms.ToTensor()
#data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(30,resample=False, expand=False, center=None),transforms.ToTensor()])

# Download training and test sets of FAshion MNIST dataset 
train_data = FashionMNIST(root='./data', train=True,
                                   download=True, transform=data_transform)

test_data = FashionMNIST(root='./data', train=False,
                                  download=True, transform=data_transform)

# Dataloader to load data in batches

batch_size = 30

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print('Done Loading data')

# Image dataset classes 
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Define CNN network

class Net(nn.Module):
    # Network Architecture
    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input channel, 20 output channel, 3*3 square kernel fo convolution
        self.conv1 = nn.Conv2d(1, 10, 3) 	# output image size = 10,26,26 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(10)
	
	# 10 input channel, 60 output channel, 3*3 square kernel for convolution
        self.conv2 = nn.Conv2d(10, 60, 3)    	# output image size = 60,11,11  
        self.conv2_bn = nn.BatchNorm2d(60)
        self.pool2 = nn.MaxPool2d(2, 2)		# output image size = 60,5,5
        
        self.fc1 = nn.Linear(60*5*5,600)
        self.drop = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(600,10)
        
    # Network feedforwad behaviour
    def forward(self, x):
        # 2 convolution layers with max pooling , batch normalization (to avoid overfit)
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.pool2(x)
	# Flatten max pool layer output to input to dense layer 
        x = x.view(x.size(0), -1)
	# Two dense layers with drop out layer action after the first dense layer (to avoid overfit)  
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # Output probabilites of each of the 10 class
        return F.log_softmax(x,dim = 1)

# Instantiate the CNN
net = Net()
# print CNN
print(net)

# Specify loss function and optimizer function for the backward pass
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr = 0.09)

## Training the CNN
# Function to train the CNN given the number of epochs
def train(n_epochs):
    
    for epoch in range(n_epochs):  # to loop over the dataset multiple times

        running_loss = 0.0
	# batch_i keeps the count of the batch, data holds the images and labels
        for batch_i, data in enumerate(train_loader):
	
            # get the input images and their corresponding labels
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass input
            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # print loss on every 1000 batches of data training 
            running_loss += loss.item()
            if batch_i % 1000 == 999:   
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0

    print('Finished Training')

# Number of epochs to train 
n_epochs = 20 
# Train
train(n_epochs)

## Test the trained Network
# Initialize tensors and lists to keep track of test loss and accuracy
test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# Set the module to evaluation mode
net.eval() 
for batch_i, data in enumerate(test_loader):
    
    # Similar to train part load test data and keep track of loss
    inputs, labels = data
    outputs = net(inputs)
    loss = criterion(outputs, labels)
            
    # update average test loss 
    test_loss += ( torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss)
    
    # get the predicted class from the maximum value in the output of class scores
    _, predicted = torch.max(outputs.data, 1)
    
    # compare predictions to true label
    correct = np.squeeze(predicted.eq(labels.view_as(predicted)))
    
    # calculate test accuracy for *each* object class
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
