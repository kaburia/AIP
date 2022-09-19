from predict import predict
from process_image import process_image, imshow
import matplotlib.pyplot as plt
import json 
import argparse

from train import parse_args

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



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
    parser= argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Image file to view')
    parser.add_argument('--model', default='densenet', type=str, help='Model Architecture')
    parser.add_argument('--topk', default=5, type=int, help='Top values to display')

    args = parser.parse_args()

    return args

def main():
    args= parse_args()

    if args.image:
        view(args.image, args.model, args.topk)


if __name__ == '__main__':
    main()
