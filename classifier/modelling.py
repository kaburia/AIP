import torch
from torchvision import transforms, datasets, models
from torch import nn

densenet121 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
alexnet = models.alexnet(weights='AlexNet_Weights.DEFAULT')


# Models to choose from
model2 = {'densenet': densenet121,
          'resnet': resnet18,
          'alexnet': alexnet}



def modelling(model_name, device='cpu'):
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
    return model


# Loading saved models
def saved_model(model_name):
    # '''state = torch.load(f'{model_name}.pth') ''' -- TO USE THIS FOR FUTURE MODELS
    state = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model = modelling(model_name)
    return model.load_state_dict(state)