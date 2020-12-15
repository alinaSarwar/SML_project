import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import facial_expressions_trained.model as fe_model
import torch

class MultiOutputModel(nn.Module):
    def __init__(self, n_pose_classes, n_occlusion_classes, n_expressions_classes):
        super().__init__()
        #### FOR fcial expressions model ### 

        # net = fe_model.Model(num_classes=7) #defult classes
        # checkpoint = torch.load('facial_expressions_trained/private_model_233_66.t7', map_location=torch.device('cpu'))
        # net.load_state_dict(checkpoint['net'])
        # self.base_model = torch.nn.Sequential(*(list(net.children())[:-2])) #remove conv + pooling layer
        # # for param in self.base_model.parameters():
        # #     param.requires_grad = False
        # # print(self.base_model)
        # self.pose = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=n_pose_classes, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        # self.occlusion = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=n_occlusion_classes, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        # self.expressions = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=n_expressions_classes, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        #### FOR fcial expressions model ### 
        #### FOR RESNET50 ### 
        # self.base_model = models.resnet50(pretrained=True)
        # self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1]))
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        # # create separate classifiers for our outputs
        # self.pose = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=n_pose_classes)
        # )
        # self.occlusion = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=n_occlusion_classes)
        # )
        # self.expressions = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=n_expressions_classes)
        # )
        #### FOR VGG19 
        self.base_model = models.vgg19(pretrained=True).features
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.pose = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_pose_classes, bias=True)
        )
        self.occlusion = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_occlusion_classes, bias=True)
        )
        self.expressions = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_expressions_classes, bias=True)
        )
        #### FOR VGG19 

        #FOR MOBILENETv2
        # self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier
        # last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier
        # so, let's do the spatial averaging: reduce width and height to 1
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        # self.pose = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=n_pose_classes)
        # )
        # self.occlusion = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=n_occlusion_classes)
        # )
        # self.expressions = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=last_channel, out_features=n_expressions_classes)
        # )
        #FOR MOBILENETv2
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        # # x = torch.flatten(x, 1)
        # # print("---- ", x.shape)
        # return {
        #     'pose': self.pose(x).squeeze(),
        #     'occlusion': self.occlusion(x).squeeze(),
        #     'expressions': self.expressions(x).squeeze()
        # }
        x = torch.flatten(x, 1)
        # print("---- ", x.shape)
        return {
            'pose': self.pose(x),
            'occlusion': self.occlusion(x),
            'expressions': self.expressions(x)
        }

    def get_loss(self, net_output, ground_truth):
        pose_loss = F.cross_entropy(net_output['pose'], ground_truth['pose_labels'])
        occlusion_loss = F.cross_entropy(net_output['occlusion'], ground_truth['occlusion_labels'])
        expressions_loss = F.cross_entropy(net_output['expressions'], ground_truth['expression_labels'])
        loss = pose_loss + occlusion_loss + expressions_loss
        # loss = pose_loss
        return loss, {'pose': pose_loss, 'occlusion': occlusion_loss, 'expressions': expressions_loss}
