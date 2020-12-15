import argparse
import os
import warnings

import matplotlib.pyplot as plt
# plt.set_cmap('gray')
# plt.rcParams['image.cmap'] = 'gray'
        
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset_vgg import FaceDataset, AttributesDataset, mean, std
from model_vgg import MultiOutputModel
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
        accuracy_age = 0
        accuracy_gender = 0
        accuracy_ethnicity = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_age, batch_accuracy_gender, batch_accuracy_ethnicity = \
                calculate_metrics(output, target_labels)

            accuracy_age += batch_accuracy_age
            accuracy_gender += batch_accuracy_gender
            accuracy_ethnicity += batch_accuracy_ethnicity

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_age /= n_samples
    accuracy_gender /= n_samples
    accuracy_ethnicity /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, age: {:.4f}, gender: {:.4f}, ethnicity: {:.4f}\n".format(
        avg_loss, accuracy_age, accuracy_gender, accuracy_ethnicity))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_accuracy_age', accuracy_age, iteration)
    logger.add_scalar('val_accuracy_gender', accuracy_gender, iteration)
    logger.add_scalar('val_accuracy_ethnicity', accuracy_ethnicity, iteration)

    model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_age_all = []
    gt_gender_all = []
    gt_ethnicity_all = []
    predicted_age_all = []
    predicted_gender_all = []
    predicted_ethnicity_all = []

    accuracy_age = 0
    accuracy_gender = 0
    accuracy_ethnicity = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_ages = batch['labels']['age_labels']
            gt_genders = batch['labels']['gender_labels']
            gt_ethnicitys = batch['labels']['ethnicity_labels']
            output = model(img.to(device))

            batch_accuracy_age, batch_accuracy_gender, batch_accuracy_ethnicity = \
                calculate_metrics(output, batch['labels'])
            accuracy_age += batch_accuracy_age
            accuracy_gender += batch_accuracy_gender
            accuracy_ethnicity += batch_accuracy_ethnicity

            # get the most confident prediction for each image
            _, predicted_ages = output['age'].cpu().max(1)
            _, predicted_genders = output['gender'].cpu().max(1)
            _, predicted_ethnicitys = output['ethnicity'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_age = attributes.age_id_to_name[predicted_ages[i].item()]
                predicted_gender = attributes.gender_id_to_name[predicted_genders[i].item()]
                predicted_ethnicity = attributes.ethnicity_id_to_name[predicted_ethnicitys[i].item()]

                gt_age = attributes.age_id_to_name[gt_ages[i].item()]
                gt_gender = attributes.gender_id_to_name[gt_genders[i].item()]
                gt_ethnicity = attributes.ethnicity_id_to_name[gt_ethnicitys[i].item()]

                gt_age_all.append(gt_age)
                gt_gender_all.append(gt_gender)
                gt_ethnicity_all.append(gt_ethnicity)

                predicted_age_all.append(predicted_age)
                predicted_gender_all.append(predicted_gender)
                predicted_ethnicity_all.append(predicted_ethnicity)

                imgs.append(image)
                labels.append("{} {} {}".format(predicted_gender, predicted_ethnicity, predicted_age))
                gt_labels.append("{} {} {}".format(gt_gender, gt_ethnicity, gt_age))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\nage: {:.4f}, gender: {:.4f}, ethnicity: {:.4f}".format(
            accuracy_age / n_samples,
            accuracy_gender / n_samples,
            accuracy_ethnicity / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # age
        cn_matrix = confusion_matrix(
            y_true=gt_age_all,
            y_pred=predicted_age_all,
            labels=attributes.age_labels,
            normalize='all')
        print("Age confusion matrix")
        print(attributes.age_labels)
        print(confusion_matrix(
            y_true=gt_age_all,
            y_pred=predicted_age_all,
            labels=attributes.age_labels))
        print()
        
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.age_labels).plot(
            include_values=True, xticks_rotation='vertical',cmap='Blues')
        plt.title("age")
        plt.set_cmap('gray')

        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attributes.gender_labels,
            normalize='all')
        print("Gender confusion matrix")
        print(attributes.gender_labels)
        print(confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attributes.gender_labels))
        print()
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.gender_labels).plot(
            xticks_rotation='horizontal',cmap='Blues')
        plt.title("gender")
        plt.set_cmap('gray')

        plt.tight_layout()
        plt.show()

        # Uncomment code below to see the ethnicity confusion matrix (it may be too big to display)
        cn_matrix = confusion_matrix(
            y_true=gt_ethnicity_all,
            y_pred=predicted_ethnicity_all,
            labels=attributes.ethnicity_labels,
            normalize='all')
        print("Ethnicity confusion matrix")
        print(attributes.ethnicity_labels)
        print(confusion_matrix(
            y_true=gt_ethnicity_all,
            y_pred=predicted_ethnicity_all,
            labels=attributes.ethnicity_labels))
        print()

        plt.rcParams.update({'font.size': 5})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=attributes.ethnicity_labels).plot(
            include_values=True, xticks_rotation='vertical',cmap='Blues')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("ethnicity types")
        plt.set_cmap('gray')

        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(40, 40))#,gridspec_kw = {'height_ratios':[15,15]})
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0, fontsize=15)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target):
    _, predicted_age = output['age'].cpu().max(1)
    gt_age = target['age_labels'].cpu()

    _, predicted_gender = output['gender'].cpu().max(1)
    gt_gender = target['gender_labels'].cpu()

    _, predicted_ethnicity = output['ethnicity'].cpu().max(1)
    gt_ethnicity = target['ethnicity_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_age = balanced_accuracy_score(y_true=gt_age.numpy(), y_pred=predicted_age.numpy())
        accuracy_gender = balanced_accuracy_score(y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
        accuracy_ethnicity = balanced_accuracy_score(y_true=gt_ethnicity.numpy(), y_pred=predicted_ethnicity.numpy())

    return accuracy_age, accuracy_gender, accuracy_ethnicity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/expUTK_vgg/2020-12-03_14-22/checkpoint-000050.pth", help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./data/UTKFace/newattributes.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((40,50)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # test_dataset = FaceDataset('./data/IMFDB_selected/val.csv', attributes, val_transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    val_dataset = FaceDataset('data/UTKFace/newval.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    model = MultiOutputModel(n_age_classes=attributes.num_age, n_gender_classes=attributes.num_gender,
                             n_ethnicity_classes=attributes.num_ethnicity).to(device)

    # Visualization of the trained model
    visualize_grid(model, val_dataloader, attributes, device, checkpoint=args.checkpoint, show_cn_matrices=True)
