import argparse
import csv
import os

import numpy as np


def save_csv(data, path, fieldnames=['image_path', 'pose', 'expressions', 'gender', 'occlusion']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('--input', type=str, default='../IMFDB_final', help="Path to the dataset")
    parser.add_argument('--output', type=str, default='../output_imfdb', help="Path to the working folder")

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output

    actors = os.listdir(input_folder)
    
    all_data = []
        
    for actor in actors:
        actor_path = os.path.join(input_folder, actor)

        if os.path.isfile(actor_path):
                continue

        for movie in os.listdir(actor_path):
            
            movie_path = os.path.join(actor_path, movie)

            if os.path.isfile(movie_path):
                continue

            txt_file = [f for f in os.listdir(movie_path) if os.path.isfile(os.path.join(movie_path, f)) and '.txt' in f]
            
            if txt_file == []: # some movies have missing txt files
                continue
            
            fob = open(os.path.join(movie_path, txt_file[0]), 'r').readlines()
            
            for img in fob:
                
                annotation = img.strip().split('\t')
                if annotation == [''] or annotation == ['\ufeff']:
                    continue

                if 'MALE' in annotation or 'FEMALE' in annotation: # some annotations dont contain class labels
                        
                    search_string = os.listdir(os.path.join(movie_path, 'images'))[0].split('_')[0]

                    jpgs = [img for img in annotation if '.jpg' in img]

                    if len(jpgs) != 0: # some annotations have 'null' image names

                        img = [j for j in jpgs if search_string in j]
                        
                        if len(img) != 1: # mismatching filenames in annotation and actual image file - Soundarya
                            continue

                        else:
                            
                            makeup, pose, age, illumination, occlusion, expr, gender = annotation[-1], annotation[-2], annotation[-3], annotation[-4], annotation[-5], annotation[-6], annotation[-7]
                            
                            img_path = os.path.join(movie_path, 'images', img[0])
                            
                            if occlusion not in ['GLASSES', 'BEARD', 'ORNAMENTS', 'HAIR', 'HAND', 'NONE', 'OTHERS']: # some annotations have missing ornament class
                                print(movie_path, img)
                                continue
                            all_data.append([img_path, pose, expr, gender, occlusion])

        print(len(all_data))

    # set the seed of the random numbers generator, so we can reproduce the results later
    np.random.seed(42)

    # construct a Numpy array from the list
    all_data = np.asarray(all_data)
    # split the data into train/val and save them as csv files
    train_data = all_data[:24169,:]
    val_data = all_data[24169:,:]

    
    save_csv(train_data, os.path.join(output_folder, 'train.csv'))
    save_csv(val_data, os.path.join(output_folder, 'val.csv'))