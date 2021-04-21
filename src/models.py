"""

  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  SPDX-License-Identifier: MIT-0
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvg2D(nn.Module):
    """ Global averaging layer """
    
    def __call__(self, tensor):
        if len(tensor.shape) != 4:
            raise Exception('tensor must be rank of 4')
            
        return tensor.mean([2, 3]).unsqueeze(-1).unsqueeze(-1)

class Classifier(nn.Module):
    """ CNN classifier """
    
    def __init__(self, n_classes, h_dim=512):
        super(Classifier, self).__init__()
        
        self.h_dim = h_dim
        
        # (1, 128, 251)
        self.encode = nn.Sequential(
            nn.Conv2d(1, 64, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 128, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 256, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, h_dim, 4, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim),
            GlobalAvg2D()
        )

        self.out = nn.Linear(h_dim, n_classes, bias=True)
        
    def forward(self, x):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        return self.out(h)