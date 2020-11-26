import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path):
        pose_labels = []
        occlusion_labels = []
        expression_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose_labels.append(row['pose'])
                occlusion_labels.append(row['occlusion'])
                expression_labels.append(row['expressions'])

        self.pose_labels = np.unique(pose_labels)
        self.occlusion_labels = np.unique(occlusion_labels)
        self.expression_labels = np.unique(expression_labels)

        self.num_poses = len(self.pose_labels)
        self.num_occlusion = len(self.occlusion_labels)
        self.num_expressions = len(self.expression_labels)

        self.pose_id_to_name = dict(zip(range(len(self.pose_labels)), self.pose_labels))
        self.pose_name_to_id = dict(zip(self.pose_labels, range(len(self.pose_labels))))

        self.occlusion_id_to_name = dict(zip(range(len(self.occlusion_labels)), self.occlusion_labels))
        self.occlusion_name_to_id = dict(zip(self.occlusion_labels, range(len(self.occlusion_labels))))

        self.expression_id_to_name = dict(zip(range(len(self.expression_labels)), self.expression_labels))
        self.expression_name_to_id = dict(zip(self.expression_labels, range(len(self.expression_labels))))


class FaceDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.pose_labels = []
        self.occlusion_labels = []
        self.expression_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.pose_labels.append(self.attr.pose_name_to_id[row['pose']])
                self.occlusion_labels.append(self.attr.occlusion_name_to_id[row['occlusion']])
                self.expression_labels.append(self.attr.expression_name_to_id[row['expressions']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        #img = Image.open(img_path)
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        # img = img.transpose((2,0,1))
        # print(img_path, img.shape)
        # img = img.reshape(120,128)
        img = Image.fromarray(img)
        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'pose_labels': self.pose_labels[idx],
                'occlusion_labels': self.occlusion_labels[idx],
                'expression_labels': self.expression_labels[idx]
            }
        }
        return dict_data
