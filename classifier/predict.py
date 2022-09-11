from data import transforming
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import json

from modelling import saved_model
from process_image import process_image, imshow




with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



def predict(image, model_name, top=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = saved_model(model_name)
    # model.eval()
    imag = process_image(image).unsqueeze(dim=0)

    with torch.no_grad():
        # model.eval()
        output = model.forward(imag)

    ps = torch.exp(output)
    probs,classes = ps.topk(top, dim=1)
    # print(classes)

    # equals = top_class == 

    return classes


# TODO: Display an image along with the top 5 classes
def view(image, model_name, topk=5):
    # model = saved_model(model_name)

    classes = predict(image, model_name) 
    imag = process_image(image) 
      
    
    clas_im = []
    for top_class in classes.numpy()[0]:
        clas_im.append(cat_to_name[str(top_class+1)])
    plt.subplots()
    x = plt.barh(clas_im, width=0.000001)
    image = imshow(imag)
    plt.show()    






def parse():
    pass

def main():
    pass

if __name__ == '__main__':
    # main()
    view('flower/test/1/image_06743.jpg', 'densenet')



'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''


# Images only work with RGB