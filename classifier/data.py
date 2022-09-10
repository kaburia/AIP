import torch
from torchvision import models, transforms, datasets
from torch import nn




# Load train data
def trainloader(img_path):
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(img_path + '/train', transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    return train_loader

# Train validation data


def val_loader(img_path):
    test_transforms  = transforms.Compose((transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])))

    val_dataset = datasets.ImageFolder(img_path + '/valid', transform=test_transforms)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    return valloader


def testloader(img_path):
    test_transforms  = transforms.Compose((transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])))

    test_dataset = datasets.ImageFolder(img_path + '/test', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return test_loader

