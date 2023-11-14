import argparse  
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import json
from PIL import Image
import numpy as np

def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory of the data')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    #Loading the data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'training_sets': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation_sets': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testing_sets': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    image_datasets = {
        "train_set": datasets.ImageFolder(train_dir, transform=data_transforms["training_sets"]),
        "val_set": datasets.ImageFolder(valid_dir, transform=data_transforms["validation_sets"]),
        "test_set": datasets.ImageFolder(test_dir, transform=data_transforms["testing_sets"])
    }


    dataloaders = {
        "train_set_": torch.utils.data.DataLoader(image_datasets["train_set"], batch_size=32, shuffle=True),
        "val_set_": torch.utils.data.DataLoader(image_datasets["val_set"], batch_size=32, shuffle=True),
        "test_set_": torch.utils.data.DataLoader(image_datasets["test_set"], batch_size=32, shuffle=False)
    }

    #Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    #Building and training the classifier

    # Defining a dictionary that maps architectures to their output sizes
    arch_output_sizes = {
        'vgg16': 25088,
        'alexnet': 9216,
        'resnet': 2048,
        'densenet': 1024
    }

    # Choosing the architecture
    if args.arch in arch_output_sizes:
        model = getattr(models, args.arch)(pretrained=True)
    else:
        print(f"Unknown model architecture: {args.arch}")
        print("Please choose from vgg16, alexnet, resnet, densenet")
        return

    # Getting the output size for the chosen architecture
    output_size = arch_output_sizes[args.arch]


    classifier_ = nn.Sequential(nn.Linear(output_size, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 102),
                                nn.LogSoftmax(dim=1))

    model.classifier = classifier_


 
    optimizer_ = optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion_ = nn.NLLLoss()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

 
    epochs = 5
    steps = 0
    print_every = 100

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders["train_set_"]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_.zero_grad()

            log_probabilities = model.forward(inputs)
            loss_value = criterion_(log_probabilities, labels)
            loss_value.backward()
            optimizer_.step()

            running_loss += loss_value.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["val_set_"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_probabilities = model.forward(inputs)
                        batch_loss = criterion_(log_probabilities, labels)

                        valid_loss += batch_loss.item()

                        probabilities = torch.exp(log_probabilities)
                        top_probabilities, top_classes = probabilities.topk(1, dim=1)
                        equals = top_classes == labels.view(*top_classes.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['val_set_']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['val_set_']):.3f}")
                running_loss = 0
                model.train()

        torch.cuda.empty_cache()  


  
    # Set the class_to_idx attribute
    model.class_to_idx = image_datasets['train_set'].class_to_idx

    # Save the checkpoint
    checkpoint = {
        'input_size': arch_output_sizes[args.arch],  # Use the output size of the chosen architecture
        'output_size': 102,
        'epochs': epochs,
        'batch_size': 64,
        'model': args.arch,  # Save the name of the chosen architecture
        'classifier': classifier_,
        'optimizer': optimizer_.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')


if __name__ == "__main__":
    main()


# References:
# https://github.com/lynguyenminh/image-classification-pytorch
# https://github.com/bentrevett/pytorch-image-classification
# https://github.com/hysts/pytorch_image_classification
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://reason.town/image-classification-with-pytorch
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# https://github.com/pytorch/ignite
# https://github.com/devforfu/loop
# https://github.com/glassroom/torch_train_test_loop
# https://gist.github.com/AStoeffelbauer/d67e14f8d6084f8805eefeb47e3c7e1b
# https://machinelearningmastery.com/creating-a-training-loop-for-pytorch-models/
# https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/what_is_pytorch.ipynb
