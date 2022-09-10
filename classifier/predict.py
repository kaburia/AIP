from data import transforming
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load a trained model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transforming = transforming()
    
    img = Image.open(image)

    return transforming(img)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path).unsqueeze(dim=0)

    with torch.no_grad():
        output = model.forward(image)

    ps = torch.exp(output)
    ps = ps
    classes = ps.topk(topk, dim=1)
    # equals = top_class == 

    return classes
    # TODO: Implement the code to predict the class from an image file

# TODO: Display an image along with the top 5 classes

def view(image_path, model, topk=5):

    probs, classes = predict(image_path, model) 
    image = process_image(image_path) 
      
    
    clas_im = []
    for top_class in classes.numpy()[0]:
        clas_im.append(cat_to_name[str(top_class)])
    plt.subplots()
    x = plt.barh(clas_im, width=0.000001)
    image = imshow(image)
    return plt.show()    


def parse():
    pass

def main():
    pass

if __name__ == '__main__':
    main()



'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''