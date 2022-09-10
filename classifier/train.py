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
import argparse

from data import trainloader, val_loader, testloader


densenet121 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')

# Models to choose from
model2 = {'densenet': densenet121,
          'resnet': resnet18,
          'alexnet': alexnet}



   
# Train a model
def train(img_path, model_name, epochs=1, learning_rate=0.03, device='cpu'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use a pretrained model
    model = model2[model_name]
    
    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    if model_name.lower() == 'alexnet':
        model.classifier = nn.Sequential(nn.Linear(9216, 4896),
                                    nn.ReLU(),
                                    nn.Linear(4896, 2448),
                                    nn.ReLU(),
                                    nn.Linear(2448,102),
                                    nn.LogSoftmax(dim=1))
    elif model_name.lower() == 'densenet':
        model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256,102),
                                 nn.LogSoftmax(dim=1))
    elif model_name.lower() == 'resnet':
        model.fc = nn.Sequential(nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256,102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device);

    
    steps = 0
    running_loss = 0
    print_every = 5

    print(f'Beginning Training:\nDirectory: {img_path} Model: {model_name}, Epochs: {epochs}, Device: {device}')
    for epoch in range(epochs):
        for inputs, labels in trainloader(img_path):
            steps += 1
            optimizer.zero_grad()

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model(inputs)
            loss = criterion(logps, labels)            
            
            loss.backward()
            # print('training loss: {}'.format(loss.item()))
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader(img_path):
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

    # Saving the model with the model name
    torch.save(model.state_dict(), f'{model_name}.pth')


    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='flower/', type=str, help='path to flower images')
    parser.add_argument('--arch', default='densenet', type=str, help='Model Architecture')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.03, type=int, help='Set Learning rate')
    parser.add_argument('--gpu', default='cpu', type=str, help='Device training from GPU/CPU')

    # Given a choice of input of directory
    parser.add_argument('--in_dir', type=str, help='The input directory to feed into the model')
    parser.add_argument('--out_dir', type=str, help='The output to save the model')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()


    if args.arch or args.dir:  
        if args.epochs or args.learning_rate or args.gpu:
            train(args.dir, args.arch, learning_rate=args.learning_rate, epochs=args.epochs, device=args.gpu)
        # train(args.dir, args.arch)

    # if args.gpu:
        # train(args.arch, device=args.gpu)

    # if args.dir:


if __name__ == '__main__':
    main()



# Set directory to save checkpoints
# Choose a specific directory to perform Training on
# Create a function to obtain the path of the input directory
# Check the format of the images
# If scattered create labels and group to pass into ImageFolder easily