import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import torch

class MultiOutputModel(nn.Module):
    def __init__(self, n_age_classes, n_gender_classes, n_ethnicity_classes):
        super().__init__()
        #### FOR VGG19 
        self.base_model = models.vgg19(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.age = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_age_classes, bias=True)
        )
        self.gender = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_gender_classes, bias=True)
        )
        self.ethnicity = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_ethnicity_classes, bias=True)
        )
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return {
            'age': self.age(x),
            'gender': self.gender(x),
            'ethnicity': self.ethnicity(x)
        }

        #### FOR VGG19

    def get_loss(self, net_output, ground_truth):
        age_loss = F.cross_entropy(net_output['age'], ground_truth['age_labels'])
        gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
        ethnicity_loss = F.cross_entropy(net_output['ethnicity'], ground_truth['ethnicity_labels'])
        loss = age_loss + gender_loss + ethnicity_loss
        # loss = age_loss
        return loss, {'age': age_loss, 'gender': gender_loss, 'ethnicity': ethnicity_loss}
