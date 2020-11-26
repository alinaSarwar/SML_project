import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import FaceDataset, AttributesDataset, mean, std
from model import MultiOutputModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_pose = 0
        accuracy_occlusion = 0
        accuracy_expression = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_pose, batch_accuracy_occlusion, batch_accuracy_expression = \
                calculate_metrics(output, target_labels)

            accuracy_pose += batch_accuracy_pose
            accuracy_occlusion += batch_accuracy_occlusion
            accuracy_expression += batch_accuracy_expression

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_pose /= n_samples
    accuracy_occlusion /= n_samples
    accuracy_expression /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, pose: {:.4f}, occlusion: {:.4f}, expression: {:.4f}\n".format(
        avg_loss, accuracy_pose, accuracy_occlusion, accuracy_expression))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_pose', accuracy_pose, iteration)
    logger.add_scalar('val_accuracy_occlusion', accuracy_occlusion, iteration)
    logger.add_scalar('val_accuracy_expression', accuracy_expression, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_pose_all = []
    gt_occlusion_all = []
    gt_expression_all = []
    predicted_pose_all = []
    predicted_occlusion_all = []
    predicted_expression_all = []

    accuracy_pose = 0
    accuracy_occlusion = 0
    accuracy_expression = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_poses = batch['labels']['pose_labels']
            gt_occlusions = batch['labels']['occlusion_labels']
            gt_expressions = batch['labels']['expression_labels']
            output = model(img.to(device))

            batch_accuracy_pose, batch_accuracy_occlusion, batch_accuracy_expression = \
                calculate_metrics(output, batch['labels'])
            accuracy_pose += batch_accuracy_pose
            accuracy_occlusion += batch_accuracy_occlusion
            accuracy_expression += batch_accuracy_expression

            # get the most confident prediction for each image
            _, predicted_poses = output['pose'].cpu().max(1)
            _, predicted_occlusions = output['occlusion'].cpu().max(1)
            _, predicted_expressions = output['expressions'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_pose = attributes.pose_id_to_name[predicted_poses[i].item()]
                predicted_occlusion = attributes.occlusion_id_to_name[predicted_occlusions[i].item()]
                predicted_expression = attributes.expression_id_to_name[predicted_expressions[i].item()]

                gt_pose = attributes.pose_id_to_name[gt_poses[i].item()]
                gt_occlusion = attributes.occlusion_id_to_name[gt_occlusions[i].item()]
                gt_expression = attributes.expression_id_to_name[gt_expressions[i].item()]

                gt_pose_all.append(gt_pose)
                gt_occlusion_all.append(gt_occlusion)
                gt_expression_all.append(gt_expression)

                predicted_pose_all.append(predicted_pose)
                predicted_occlusion_all.append(predicted_occlusion)
                predicted_expression_all.append(predicted_expression)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_occlusion, predicted_expression, predicted_pose))
                gt_labels.append("{}\n{}\n{}".format(gt_occlusion, gt_expression, gt_pose))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\npose: {:.4f}, occlusion: {:.4f}, expressions: {:.4f}".format(
            accuracy_pose / n_samples,
            accuracy_occlusion / n_samples,
            accuracy_expression / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # pose
        cn_matrix = confusion_matrix(
            y_true=gt_pose_all,
            y_pred=predicted_pose_all,
            labels=attributes.pose_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.pose_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("poses")
        plt.tight_layout()
        plt.show()

        # occlusion
        cn_matrix = confusion_matrix(
            y_true=gt_occlusion_all,
            y_pred=predicted_occlusion_all,
            labels=attributes.occlusion_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.occlusion_labels).plot(
            xticks_rotation='horizontal')
        plt.title("occlusions")
        plt.tight_layout()
        plt.show()

        # Uncomment code below to see the expression confusion matrix (it may be too big to display)
        cn_matrix = confusion_matrix(
            y_true=gt_expression_all,
            y_pred=predicted_expression_all,
            labels=attributes.expression_labels,
            normalize='true')
        plt.rcParams.update({'font.size': 1.8})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(cn_matrix, attributes.expression_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("expression types")
        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_pose = output['pose'].cpu().max(1)
    gt_pose = target['pose_labels'].cpu()

    _, predicted_occlusion = output['occlusion'].cpu().max(1)
    gt_occlusion = target['occlusion_labels'].cpu()

    _, predicted_expression = output['expressions'].cpu().max(1)
    gt_expression = target['expression_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_pose = balanced_accuracy_score(y_true=gt_pose.numpy(), y_pred=predicted_pose.numpy())
        accuracy_occlusion = balanced_accuracy_score(y_true=gt_occlusion.numpy(), y_pred=predicted_occlusion.numpy())
        accuracy_expression = balanced_accuracy_score(y_true=gt_expression.numpy(), y_pred=predicted_expression.numpy())

    return accuracy_pose, accuracy_occlusion, accuracy_expression


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./fashion-product-images/styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_pose_classes=attributes.num_poses, n_occlusion_classes=attributes.num_occlusions,
                             n_expression_classes=attributes.num_expressions).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
