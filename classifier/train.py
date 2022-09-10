# Required libraries
from random import shuffle
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Load and transform images
def load_image(img_path, method):
    if method == 'train':
        train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        train_dataset = datasets.ImageFolder(img_path + '/train', transform=train_transform)

        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    elif method == 'valid' or method == 'test':
        test_transforms  = transforms.Compose((transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])))
        if method == 'valid':
            val_dataset = datasets.ImageFolder(img_path + '/valid', transform=test_transforms)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

        else:
            test_dataset = datasets.ImageFolder(img_path + '/test', transform=test_transforms)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    
'''

'''    
# Train a model
def train(trainloader, model, epochs):
    pass
