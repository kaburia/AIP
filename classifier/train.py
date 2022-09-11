# Required libraries
import torch
from torch import  nn
from torch.optim import Adam
import torch.nn.functional as F
import argparse

from data import trainloader, val_loader, testloader
from modelling import modelling


   
# Train a model
def train(img_path, model_name, epochs=1, learning_rate=0.03, device='cpu'):
    
    model = modelling(model_name)
    trainload = trainloader(img_path)
    val_load = val_loader(img_path)
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5

    print(f'Beginning Training:\nDirectory: {img_path} Model: {model_name}, Epochs: {epochs}, Device: {device}')
    # epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainload:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_load:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(val_load):.3f}.. "
                    f"Test accuracy: {accuracy/len(val_load):.3f}")
                running_loss = 0
                model.train()

    # Saving the model with the model name
    torch.save(model.state_dict(), f'{model_name}.pth')


    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='flower/', type=str, help='path to flower images')
    parser.add_argument('--arch', default='densenet', type=str, help='Model Architecture')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.03, type=int, help='Set Learning rate')
    parser.add_argument('--gpu', default='cpu', type=str, help='Device training from GPU/CPU')

    # Given a choice of input of directory
    parser.add_argument('--in_dir', type=str, help='The input directory to feed into the model')
    parser.add_argument('--out_dir', type=str, help='The output to save the model')

    args = parser.parse_args()

    return args




def main():
    args = parse_args()

    # if args.dir:
    #     print(run_tests(args.dir))


    if args.arch or args.dir:
        if args.epochs or args.learning_rate or args.gpu:
            train(args.dir, args.arch, learning_rate=args.learning_rate, epochs=args.epochs, device=args.gpu)

if __name__ == '__main__':
    main()



# Set directory to save checkpoints
# Choose a specific directory to perform Training on
# Create a function to obtain the path of the input directory
# Check the format of the images
# If scattered create labels and group to pass into ImageFolder easily

# Check the imports from data