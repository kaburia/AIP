import json
import os


flower_names = {}
train_dir = 'flower/train'

def labels():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    for no, names in enumerate(sorted(os.listdir(train_dir))):
        flower_names[str(no)] = names
    
    return flower_names


