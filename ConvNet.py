import time
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        #First Convolution Kayer
        #input size (28,28), output size = (24,24)
        self.conv1 = nn.Conv2d(1,20,5)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)

        #Second Convolution Layer
        #input size (12,12), output_size = (8,8)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2)
        self.reLU = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout(0.5)

        #Affine operations
        #Single Fully Connected layer with 
        self.fc = nn.Linear(in_features=28*28,out_features=100)
        
        self.fc1 = nn.Linear(in_features = 40*4*4, out_features = 100)
        self.sig = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        #Model #4 FC layers
        self.fc1_mod4 =nn.Linear(in_features=40*4*4,out_features=248)
        self.fc2_mod4 =nn.Linear(in_features=248,out_features=100)

        # Model 5 FC Layers
        self.fc1_mod5 = nn.Linear(in_features=40*4*4, out_features=1000)
        self.fc2_mod5 = nn.Linear(in_features=1000,out_features = 248)
        self.fc3_mod5 = nn.Linear(in_features=248,out_features = 100)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #return NotImplementedError()

        #Conv Layer #1
       
        #print(Tensor.size(X))
        #X = X.view(10,16*4*4)
        #Flatten the shape of the tensor to (batch_size, num_features)
        X = X.view(X.size(0),-1)
        X = self.fc(X)
        X = self.sig(X)
        X = self.fc2(X)
      
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        
        #Conv Layer #1
        X = self.conv1(X)
        X = self.MaxPool1(X)
        #Conv Layer #2
        X = self.conv2(X)
        X = self.MaxPool2(X)

        #print(Tensor.size(X))
        #X = X.view(10,40*4*4)
        #Flatten the shape of the tensor to (batch_size, num_features)
        X = X.view(X.size(0),-1)
        X = self.fc1(X)
        X = self.sig(X)
        X = self.fc2(X) 
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
                #Conv Layer #1
        X = self.conv1(X)
        X = self.reLU(X)
        X = self.MaxPool1(X)
        #Conv Layer #2
        X = self.conv2(X)
        X= self.reLU(X)
        X = self.MaxPool2(X)

        #print(Tensor.size(X))
        #X = X.view(10,40*4*4)
        #Flatten the shape of the tensor to (batch_size, num_features)
        X = X.view(X.size(0),-1)
        X = self.fc1(X)
        X = self.reLU(X)
        X = self.fc2(X)

       

        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        #Conv Layer #1
        X = self.conv1(X)
        X = self.reLU(X)
        X = self.MaxPool1(X)
        #Conv Layer #2
        X = self.conv2(X)
        X= self.reLU(X)
        X = self.MaxPool2(X)

        #print(Tensor.size(X))
        #X = X.view(10,40*4*4)
        #Flatten the shape of the tensor to (batch_size, num_features)
        X = X.view(X.size(0),-1)
        X = self.fc1_mod4(X)
        X = self.reLU(X)
        X = self.fc2_mod4(X)
        X - self.reLU(X)
        X = self.fc2(X)

        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
                #Conv Layer #1
        X = self.conv1(X)
        X = self.reLU(X)
        X = self.MaxPool1(X)
        #Conv Layer #2
        X = self.conv2(X)
        X= self.reLU(X)
        X = self.MaxPool2(X)
        X = self.dropout1(X)

        #Flatten the shape of the tensor to (batch_size, num_features)
        X = X.view(X.size(0),-1)
        X = self.fc1_mod5(X)
        X = self.reLU(X)
        X = self.dropout1(X)
        X = self.fc2_mod5(X)
        X = self.reLU(X)
        X = self.dropout1(X)
        X = self.fc3_mod5(X)
        X = self.reLU(X)
        X = self.dropout1(X)
        X = self.fc2(X)

        return X
    
    
