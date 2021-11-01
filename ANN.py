import torch
import torch.nn as nn

class ANN(nn.module):
    def __init__(self):
        super.__init__()
        self.flatten=nn.flatten()
        self.dense_layers= nn.Sequential(
            nn.Linear(64*313,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )
        self.softmax=nn.softmax(dim=1)

    def forward(self,x):
        x=self.flatten(x)
        x=self.dense_layers(x)
        prediction=self.softmax(x)
        return prediction
    