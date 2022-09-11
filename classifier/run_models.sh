# Run the various models
python train.py --arch resnet --dir flower/ --epochs 10 --gpu gpu
python train.py --arch densnet --dir flower/ --epochs 10 --gpu gpu
python train.py --arch alexnet --dir flower/ --epochs 10 --gpu gpu



# Get the different accuracies and compare model with the highest one