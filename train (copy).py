import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset import FaceDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from test import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='data/UTKFace/newattributes.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 300
    batch_size = 128
    lr = 0.0001
    num_workers = 8  # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.Resize((48,48)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2),
        #                         shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FaceDataset('data/UTKFace/newtrain.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FaceDataset('data/UTKFace/newval.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiOutputModel(n_pose_classes=attributes.num_poses,
                             n_occlusion_classes=attributes.num_occlusion,
                             n_expressions_classes=attributes.num_expressions)\
                            .to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    

    logdir = os.path.join('./logs/exp40x80_noMeanSTD', get_cur_time())
    savedir = os.path.join('./checkpoints/exp40x80_noMeanSTD', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    visualize_grid(model, train_dataloader, attributes, device, show_cn_matrices=False, show_images=True, checkpoint=None, show_gt=True)
    # print("\nAll occlusion labels:\n", attributes.eyes_labels)
    # print("\nAll pose labels:\n", attributes.pose_labels)
    # print("\nAll expression labels:\n", attributes.expression_labels)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        for param_group in optimizer.param_groups:
            print("lr ->>>>> ", param_group['lr'])

        total_loss = 0
        accuracy_pose = 0
        accuracy_occlusion = 0
        accuracy_expression = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            #pose, occlusion, expression
            batch_accuracy_pose, batch_accuracy_occlusion, batch_accuracy_expression = \
                calculate_metrics(output, target_labels)

            accuracy_pose += batch_accuracy_pose
            accuracy_occlusion += batch_accuracy_occlusion
            accuracy_expression += batch_accuracy_expression

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, pose: {:.4f}, occlusion: {:.4f}, expressions: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_pose / n_train_samples,
            accuracy_occlusion / n_train_samples,
            accuracy_expression / n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)
