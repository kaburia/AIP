import torch

from modelling import saved_model
from process_image import process_image






def predict(image_path, model, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path).unsqueeze(dim=0)

    with torch.no_grad():
      model.eval()
      output = model.forward(image)

    ps = torch.exp(output)
    # label = labels.cpu()
    ps, classes = ps.topk(topk, dim=1)
    # equals = classes == label.view(*top_class.shape)

    return ps, classes
    # TODO: Implement the code to predict the class from an image file


# TODO: Display an image along with the top 5 classes





def parse():
    pass

def main():
    pass

if __name__ == '__main__':
    main()
    # view('flower/test/1/image_06743.jpg', 'densenet')



'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''


# Images only work with RGB