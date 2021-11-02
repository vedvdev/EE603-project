import torch
import torch.nn as nn

def conv_block(in_channels,out_channels,kernel_size=3,stride=1,padding=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=out_channels)
        )

class CNN_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=conv_block(in_channels=1,out_channels=16)
        self.max_pool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=conv_block(in_channels=16,out_channels=64)
        self.max_pool2=nn.MaxPool2d(kernel_size=2)
        self.conv3=conv_block(in_channels=64,out_channels=128)
        self.max_pool3=nn.MaxPool2d(2)
        self.conv4=conv_block(in_channels=128,out_channels=512)
        self.global_pool= nn.MaxPool2d(kernel_size=(11,42))
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(512,1024)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout=nn.Dropout(.3)
        self.linear2=nn.Linear(1024,3)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,input):
        x=self.conv1(input)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=self.max_pool2(x)
        x=self.conv3(x)
        x=self.max_pool3(x)
        x=self.conv4(x)
        x=self.global_pool(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.softmax(x)
        
        
        return x
