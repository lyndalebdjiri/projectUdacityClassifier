
<h3>AI Programming with Python Project</h3>

This project is part of Udacity’s AI Programming with Python Nanodegree program. The primary objective is to develop a Python application that can train an image classifier on a dataset and then predict new images using the trained model. Here are the key points:

- **Objective**: Develop a Python application that can train an image classifier on a dataset and then predict new images using the trained model.
- **Python Libraries and Modules**: Utilizes `argparse`, `torch`, `torchvision`, `matplotlib`, `numpy`, `seaborn`, `json`, and `PIL`.
- **Data Handling**: Uses `torchvision` for data augmentation, loading the dataset, and accessing pretrained models. `PIL` is used for handling image file formats.
- **Neural Network Creation and Optimization**: Employs `torch` and its `nn` and `optim` modules for creating the neural network and optimizing its parameters.
- **Visualization**: Leverages `matplotlib` and `seaborn` for creating visualizations and statistical data visualization.
- **Command-Line Arguments**: Uses `argparse` for parsing command-line options and arguments.
- **JSON Data**: Utilizes `json` for working with JSON data.
- **Main Function**: Includes a main function that parses command-line arguments, loads and transforms the dataset, creates a DataLoader, builds and trains the classifier, and saves the trained model. The architecture of the model (such as ‘vgg16’, ‘alexnet’, ‘resnet’, ‘densenet’) can be chosen by the user. The learning rate, number of hidden units, and number of epochs can also be set by the user. The model can be trained on a GPU if available.
- **Model Prediction**: The trained model is used to predict the class of new images, showcasing the practical application of the project.
