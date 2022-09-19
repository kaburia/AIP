import json
import os


flower_names = {}

def labels(train_dir):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    for no, names in enumerate(sorted(os.listdir(train_dir))):
        flower_names[str(no)] = names
    
    return flower_names


