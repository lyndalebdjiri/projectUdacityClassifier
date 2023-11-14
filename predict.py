import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
import json

def main():
    
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
    parser.add_argument('input', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

 
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    #Loading the checkpoint
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)

        # Get the name of the architecture from the checkpoint
        arch = checkpoint['model']

        # Create the model based on the saved architecture
        model = getattr(models, arch)(pretrained=True)

        # Set up the classifier with the correct input size
        classifier_input_size = checkpoint['input_size']
        classifier = nn.Sequential(nn.Linear(classifier_input_size, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4096, 102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier

        # Load the rest of the checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model, optimizer


    model, optimizer = load_checkpoint(args.checkpoint)

    #Image Preprocessing
    def process_image(image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        img = Image.open(image_path)
        if img.size[0] > img.size[1]:
            img.thumbnail((10000, 256))
        else:
            img.thumbnail((256, 10000))
        left_margin = (img.width-224)/2
        bottom_margin = (img.height-224)/2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        img = img.crop((left_margin, bottom_margin, right_margin, top_margin))   
        img = np.array(img)/255
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([0.229, 0.224, 0.225]) 
        img = (img - mean)/std
        img = img.transpose((2, 0, 1))
        return img
    

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    #Class Prediction  
    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Move the model to the correct device
        model.to(device)

        img = process_image(image_path)
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        # Move the input tensor to the correct device
        model_input = image_tensor.unsqueeze(0).to(device)

        probs = torch.exp(model.forward(model_input))
        top_probs, top_labs = probs.topk(topk)
        top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
        top_labs = top_labs.detach().cpu().numpy().tolist()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labs]
        top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
        return top_probs, top_labels, top_flowers



    probs, labs, flowers = predict(args.input, model, args.top_k) 

 
    print("Probabilities:", probs)
    print("Classes:", labs)

if __name__ == "__main__":
    main()

# References:
#https://github.com/pytorch/ignite
#https://github.com/Magus4450/PyTorch-TrainingLoop
#https://github.com/glassroom/torch_train_test_loop
#https://github.com/devforfu/loop
#https://gist.github.com/AStoeffelbauer/d67e14f8d6084f8805eefeb47e3c7e1b