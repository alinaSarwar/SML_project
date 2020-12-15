import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import facial_expressions_trained.model as fe_model
import torch

class MultiOutputModel(nn.Module):
    def __init__(self, n_age_classes, n_gender_classes, n_ethnicity_classes):
        super().__init__()
        #### FOR RESNET50 ### 
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1]))
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        # create separate classifiers for our outputs
        self.age = nn.Sequential( #make 4, honda, suzuki, toyota, others, 4
            nn.Linear(in_features=2048, out_features=n_age_classes)
        )
        self.gender = nn.Sequential( #color: white, black, red, grey, silver,5
            nn.Linear(in_features=2048, out_features=n_gender_classes)
        )
        self.ethnicity = nn.Sequential(#type: hatchback, sedan, suv, 3
            nn.Linear(in_features=2048, out_features=n_ethnicity_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        # print("---- ", x.shape)
        return {
            'age': self.age(x),
            'gender': self.gender(x),
            'ethnicity': self.ethnicity(x)
        }

    def get_loss(self, net_output, ground_truth):
        age_loss = F.cross_entropy(net_output['age'], ground_truth['age_labels'])
        gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
        ethnicity_loss = F.cross_entropy(net_output['ethnicity'], ground_truth['ethnicity_labels'])
        loss = age_loss + gender_loss + ethnicity_loss
        # loss = age_loss
        return loss, {'age': age_loss, 'gender': gender_loss, 'ethnicity': ethnicity_loss}
