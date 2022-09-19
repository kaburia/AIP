from predict import predict
from process_image import process_image, imshow
import matplotlib.pyplot as plt
import argparse
from modelling import saved_model, modelling


# from train import parse_args
from labels import labels


def view(image, model_name, topk=5):
    model = saved_model(model_name)
    # model = modelling(model_name)
    # model = saved_model(model_name)
    probs, classes = predict(image, model) 
    imag = process_image(image) 
      
    
    clas_im = [labels()[str(top_class)] for top_class in classes.numpy()[0]]
    prbs = [pr for pr in probs.numpy()[0]]
    x = plt.barh(clas_im, prbs, color='purple')
    image = imshow(imag)
    return plt.show() 

def parse():
    parser= argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image file to view')
    parser.add_argument('--model', default='densenet', type=str, help='Model Architecture')
    parser.add_argument('--topk', default=5, type=int, help='Top values to display')

    args = parser.parse_args()

    return args

def main():
    args= parse()

    if args.image:
        view(args.image, args.model, args.topk)


if __name__ == '__main__':
    main()
