import torch
import torch.nn as nn

class resnet_regression(nn.Module):
    def __init__(self):
        # 3 : image, 1 : bmi
        super(resnet_regression, self).__init__()

        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs,256), nn.BatchNorm1d(256),nn.Dropout(0.2),
                          nn.Linear(256,128), nn.BatchNorm1d(128),nn.Dropout(0.2),
                          nn.Linear(128,64), nn.Linear(64,1))



    def forward(self,x_image):
        return self.resnet(x_image)