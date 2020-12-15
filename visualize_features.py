import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms
from dataset import FaceDataset, AttributesDataset, mean, std
from model import MultiOutputModel

parser = argparse.ArgumentParser(description='Inference pipeline')
parser.add_argument('--checkpoint', type=str, default="checkpoints/exp40x80_noMeanSTD/2020-11-26_23-20/checkpoint-000050.pth", help="Path to the checkpoint")
parser.add_argument('--attributes_file', type=str, default='./data/scripts/newattributes.csv',
                    help="Path to the file with attributes")
parser.add_argument('--device', type=str, default='cuda',
                    help="Device: 'cuda' or 'cpu'")
parser.add_argument('-i', '--image', required=False, default="input/img.png",
help='path to image')

args = parser.parse_args()

device = torch.device("cpu")
# attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
attributes = AttributesDataset(args.attributes_file)

model = MultiOutputModel(n_pose_classes=attributes.num_poses, n_occlusion_classes=attributes.num_occlusion,
                             n_expressions_classes=attributes.num_expressions).to(device)
   
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

# args = vars(ap.parse_args())

# load the model
# model = models.vgg19(pretrained=True)
print(model)
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            # for child in model_children[i][j].children():
            child = model_children[i][j]
            if type(child) == nn.Conv2d:
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

# visualize the first conv layer filters
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(f'./outputs/filter.png')
# plt.show()

# exit(0)
# read and visualize an image
img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0).to("cuda:0")
print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter.cpu().numpy(), cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"./outputs/vgg_layer_{args['image'].split('/')[-1].split('.')[0]}_{num_layer}.jpg")
    # plt.show()
    plt.close()