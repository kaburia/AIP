# Required libraries
from random import shuffle
from re import T
import torch
from torch import  nn
from torch.optim import Adam
from torchvision import transforms, datasets, models
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

densenet121 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')

# Models to choose from
model2 = {'densenet': densenet121,
          'resnet': resnet18,
          'alexnet': alexnet}


# Load and transform images
img_path = 'flower'

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(img_path + '/train', transform=train_transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


test_transforms  = transforms.Compose((transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])))

val_dataset = datasets.ImageFolder(img_path + '/valid', transform=test_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)


test_dataset = datasets.ImageFolder(img_path + '/test', transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)


# Define neural network 
class Network(nn.Module): 
   def __init__(self, input_size, output_size): 
       super(Network, self).__init__() 
        
       self.layer1 = nn.Linear(input_size, 256) 
       self.layer2 = nn.Linear(256, 108) 
       self.layer3 = nn.Linear(108, output_size) 


   def forward(self, x): 
       x1 = F.relu(self.layer1(x)) 
       x2 = F.relu(self.layer2(x1)) 
       x3 = self.layer3(x2) 
       return x3 
 

   
# Train a model
def train(model_name, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Alexnet
    if model_name == 'alexnet':
        model = Network(4096, 102)
    elif model_name == 'densenet121':
        model = Network(1024, 102)
    # elif model_name == 'resnet18':
        # model = Network(512, 102)

    # Use a pretrained model
    model = model2[model_name]
    
    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.parameters(), lr=0.003)

    model.to(device);

    
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(val_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(val_loader):.3f}")
                running_loss = 0
                model.train()
    

# print(alexnet.classifier)
# print(densenet121.classifier)
# print(type(model['alexnet']))
# print(resnet18)

train('resnet', 1)


