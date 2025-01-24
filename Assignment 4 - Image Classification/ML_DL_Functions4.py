import numpy as np
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 3
        kernel_size = 5
        padding = padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(8*n*28*14,100)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(100,2)
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        out = self.pool1(self.relu1(self.conv1(inp)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.pool3(self.relu3(self.conv3(out)))
        out = self.pool4(self.relu4(self.conv4(out)))
        
        # Flatten for fully connected layers
        out = out.reshape(out.size(0), -1)  # Flatten to (N, *)  
        out = self.relu_fc1(self.fc1(out))
        out = self.fc2(out)  # Final output
        return out

class CNNChannel(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 8
        kernel_size = 3
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2 * n, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=2 * n, out_channels=4 * n, kernel_size=kernel_size, padding=padding)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(in_channels=4 * n, out_channels=8 * n, kernel_size=kernel_size, padding=padding)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fix input size for fc1
        self.fc1 = nn.Linear(8 * n * 14 * 14, 100)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):  # Do NOT change the signature of this function
        left_shoe = inp[:, :, :224, :]  # Top half (left shoe)
        right_shoe = inp[:, :, 224:, :]  # Bottom half (right shoe)

        # Concatenate along the channel dimension (dim=1)
        concatenated = torch.cat((left_shoe, right_shoe), dim=1)  # Shape: (N, 6, 224, 224)

        # Pass the manipulated input through the CNN layers
        out = self.pool1(self.relu1(self.conv1(concatenated)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.pool3(self.relu3(self.conv3(out)))
        out = self.pool4(self.relu4(self.conv4(out)))

        # Flatten for fully connected layers
        out = out.reshape(out.size(0), -1)  # Flatten to (N, *)
        out = self.relu_fc1(self.fc1(out))
        out = self.fc2(out)  # Final output layer

        return out
