# Deep_Learning_Assignment_2
# CS24M023
# A Deep learning programming assignment where we have to develop a Convolution Neural Network Using a pre-trained model (which is already trained on IMAGENET dataset) from scratch to classify the images in INATURALIST dataset

Convolution Neural Networks
#### Part A -  CNN from Scratch
#### Part B -  CNN using pretrained model

## Hyperparameter configurations used
 - model: 'inceptionv3','inceptionrn', 'resnet', 'xception'
 - dropout:  0, 0.2, 0.4
 - num_dense: '128, 256, 512
  
#### Training the model
 - to train the model we need to initalise the model and use fit() function to train the model
#### Evaluating the model
 - to evaluate the model just pass the test data to Predict() function it will return the predicted class labels on test data and use evaluate() method for metrics

## Functions 
#### CNN():
- num_of_filters : number of filters per each layer 
- size_of_filters : Size of Each filter
- activation_function : Activation function for each convolution layer (relu or elu or selu)
- input_size : Input image Shape 
- dense_layer_neurons : number of Neurons in Dense layer (Pre Output Layer)
- output_size : Number of Classes present in the data
- learning_rate : Learning rate of the network (default = 0.0001)
- weight_decay : weight decay value for L2 Regularization (default = 0)
- batch_normalization: bool value that whether batch normalization layer should be added or not (default = False)
- batch_size : batch size for data (default = 32)
- data_augmentation : bool value that whether data augmentation should be applied or not (default = False)
- dropout : dropout rate value for the network (default : 0)

#### Train():
 Compiles the CNN model and trains the input data and gives accuracies for validation data
 
#### Predict():
  Predicts the output on test data
  
## Hyperparameter configurations used
activation = ELU,
batch_norm = True,
batch_size = 32,
dense_layer = 512,
dropout = 0.4,
num_filters = [3,5,3,5,3],
learning_rate = 0.0001,
filters = [128,128,64,64,32],
optimizer_function = rmsprop,
weight_decay = 0.0004,
data_augmentation = True,
  
#### Training the model
 - to train the model we need to initalise the Neural network with CNN() function and Train the model using Train() method it will display validation accuracy, training accuracy 
#### Evaluating the model
 - to evaluate the model just pass the test data to Predict() function it will return the predicted class labels on test data and use evaluate() method for metrics
 
----------------------------------------------------------
# WandB Report Link: 
https://wandb.ai/cs24m023-indian-institute-of-technology-madras/Deep_Learning_Assignment_2/reports/DA6401-Assignment-2--VmlldzoxMjM1OTIzMw

# GitHub Repo Link: 
https://github.com/lokeshtalamala1/Deep_Learning_Assignment_2
