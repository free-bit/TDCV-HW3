import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
#from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

dataset_folders = ['coarse', 'fine', 'real']
class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']
dataset_types = ["train", "db", "test"]

def convert_full_batch_images(data):
    num_class, num_images, H, W, C = data.shape
    full_batch = np.reshape(data, (-1, H, W, C))
    
    labels = np.array([])
    for i in range(num_class):
        labels = np.hstack((labels, np.full(num_images, i)))

    return full_batch, labels

def convert_full_batch_poses(data):
    num_class, num_images, poses = data.shape
    full_batch = np.reshape(data, (-1, poses))
    return full_batch

def get_images(dataset):
    class_index = 0
    # iterate over all classes
    imag = list()
    for label in class_folders:
        # get filenames
        files = os.listdir('../dataset/'+dataset+'/' + label + '/')
        # filter all png images
        files = list(filter(lambda x: x.endswith('.png'), files))
        # generate empty list of size [classes, pictures of class]
        tmp = [None]*len(files)
        # iterate over files
        for file in files:
            # filter index of file
            index=int(file.split('.png')[0].split(dataset)[1])
            # save image at corresponding position
            tmp[index] = image.imread('../dataset/'+dataset+'/' + label + '/' + file)
        imag.append(tmp)
        class_index+=1
    return imag

def get_poses(dataset):
    class_index = 0
    # iterate over all classes
    poses = list()
    for label in class_folders:
        filepath = '../dataset/'+dataset+'/' + label + '/poses.txt'
        #num_images = len(list(filter(lambda x: x.endswith('.png'), os.listdir('../dataset/'+dataset+'/' + label + '/'))))
        # generate empty list of size [classes, pictures of class]
        #tmp = [None]*num_images
        pose = list()
        file = open(filepath, "r")
        modulo = 1
        #index = 0
        for line in file: 
            if modulo%2 == 0:
                tmp = line.split()
                tmp = [float(element) for element in tmp]
                pose.append(tmp)
            modulo+=1
        poses.append(pose)
        class_index+=1
    return poses

def get_datasets():
    images_real = get_images('real')
    poses_real = get_poses('real')
    images_coarse = get_images('coarse')
    poses_coarse = get_poses('coarse')
    images_fine = get_images('fine')
    poses_fine = get_poses('fine')
    
    S_db_images = images_coarse
    S_db_poses = poses_coarse
    
    S_train_images = [list(), list(), list(), list(), list()]
    S_train_poses = [list(), list(), list(), list(), list()]
    
    S_test_images = [list(), list(), list(), list(), list()]
    S_test_poses = [list(), list(), list(), list(), list()]
    
    # get indices to split real data in training and test data
    training_split_indices = list()
    filepath = '../dataset/real/training_split.txt'
    file = open(filepath, 'r')
    for line in file:
        indices = line.split(', ')
        for i in indices:
            i = int(i)
            training_split_indices.append(i)
    
    # distribute real images into train and test datasets with respect to split indices
    for i in range(len(images_real[0])):
        # image belongs to training set
        if i in training_split_indices:
            #print('yes')
            for j in range(len(images_real)):
                S_train_images[j].append(images_real[j][i])
                S_train_poses[j].append(poses_real[j][i])
        # image belongs to test set
        else:
            for j in range(len(images_real)):
                S_test_images[j].append(images_real[j][i])
                S_test_poses[j].append(poses_real[j][i])

    # append all fine images to train dataset
    num_classes = len(images_fine)
    # Take all images under i-th class from the "fine" set and extend the corresponding class under train dataset with them
    for i in range(num_classes):
        S_train_images[i].extend(images_fine[i])
        S_train_poses[i].extend(poses_fine[i])
        
    '''missing normalization of the RGB channels'''
    return np.array(S_train_images), np.array(S_train_poses), np.array(S_test_images), np.array(S_test_poses), np.array(S_db_images), np.array(S_db_poses)


def shuffle_triplets(triplets):
    tmp = list()
    num_triplets = int(len(triplets)/3)   
    for i in range(num_triplets):
        tmp.append([triplets[3*i], triplets[3*i+1], triplets[3*i+2]])   
    tmp = np.array(tmp)
    np.random.shuffle(tmp)
    tmp = np.reshape(tmp, (-1, 64, 64, 3))   
    return tmp

def generate_all_triplets(train_images, train_poses, db_images, db_poses, plot=False):
    # initialize empty lists to generate triplets
    triplet_images = []
    triplet_poses = []
    num_class, num_images = train_images.shape[0:2]
    triplet_counter = 0

    for c in range(num_class):
        for i in range(num_images):
            triplet_counter +=1
            diff_min = 10^8
            idx = 0
            # generate random indices
            #random_class = np.random.randint(0, len(train_images))
            #random_Img = np.random.randint(0, len(train_images[0]))
            #c = random_class
            #i = random_Img
            # save random anchor image of the train database
            anchor_image = train_images[c][i]
            anchor_pose = train_poses[c][i]

            # find closest image of the S_db of the same class
            for k in range(len(db_images[c])):
                # find closest pose using given formular
                diff = 2*np.arccos(np.abs(np.dot(anchor_pose, db_poses[c][k])))
                if (diff != 0.0 and diff < diff_min):
                    diff_min = diff
                    idx = k

            puller_image = db_images[c][idx]
            puller_pose = db_poses[c][idx]
            # take randomly another class
            pusher_class = (c + np.random.randint(1, len(db_images)-1))%len(db_images)
            # take randomly an image of the class
            pusher_idx = np.random.randint(0, len(db_images))
            pusher_image = db_images[pusher_class][pusher_idx]
            pusher_pose = db_poses[pusher_class][pusher_idx]
            triplet_images.extend([anchor_image, puller_image, pusher_image])
            triplet_poses.extend([anchor_pose, puller_pose, pusher_pose])

    # shuffle the triplet_images
    triplet_images = shuffle_triplets(triplet_images)
    for l in range(num_class*num_images):
            ## Plot the Anchor, Puller pusher if wanted
            if plot:
                fig = plt.figure()
                for j in range(3):
                    fig.add_subplot(1, 3, j + 1)
                    triplet_idx = (l)*3 + j
                    img = triplet_images[triplet_idx]
                    plt.imshow(img)
                plt.show()

    '''NOTE: delete anchor_image and pose of S_train_images and S_train_poses locally --> not the same pic twice'''
    # train_size * 3 x Height x Width x Channel
    return np.array(triplet_images)

class CustomDataset(Dataset):

    def __init__(self, type="train", build=False, copy_from=None):
        self.train_triplets = self.S_train_images = self.S_train_poses\
            = self.S_test_images = self.S_test_poses = self.S_test_labels\
            = self.S_db_images = self.S_db_poses = self.S_db_labels = None

        self.type = type
        if self.type not in dataset_types:
            raise "ERROR: Unknown dataset type!"

        if not copy_from:
            # Build datasets from scratch
            if build:
                print("Building the dataset...")
                
                print("Fetching images and poses...")
                self.S_train_images, self.S_train_poses, self.S_test_images, self.S_test_poses, self.S_db_images, self.S_db_poses = get_datasets()
                print("Images and poses fetched.")

                print("Generating all triplets for training...")
                self.train_triplets = generate_all_triplets(self.S_train_images, self.S_train_poses, self.S_db_images, self.S_db_poses)
                print("All triplets generated.")

                # Remove extra class dimensions from datasets, store class information on a separate array
                self.S_test_images, self.S_test_labels = convert_full_batch_images(self.S_test_images)
                self.S_test_poses = convert_full_batch_poses(self.S_test_poses)

                self.S_db_images, self.S_db_labels = convert_full_batch_images(self.S_db_images)
                self.S_db_poses = convert_full_batch_poses(self.S_db_poses)

                # Reposition channel axis according to PyTorch convention
                self.adjust_channel_axis()
                
                print("Saving dataset...")
                np.savez("data.npz", 
                        train_triplets=self.train_triplets, S_train_images=self.S_train_images, S_train_poses=self.S_train_poses, 
                        S_test_images=self.S_test_images, S_test_labels=self.S_test_labels, S_test_poses=self.S_test_poses, 
                        S_db_images=self.S_db_images, S_db_labels=self.S_db_labels, S_db_poses=self.S_db_poses)
                print("Dataset saved.")

            # Load from a file
            else:
                # NOTE: Do not assume a specific order for keys in python dicts!
                print("Loading dataset...")
                data = np.load("data.npz")
                self.train_triplets = data["train_triplets"]
                self.S_train_images = data["S_train_images"]
                self.S_train_poses = data["S_train_poses"]
                self.S_test_images = data["S_test_images"]
                self.S_test_labels = data["S_test_labels"]
                self.S_test_poses = data["S_test_poses"]
                self.S_db_images = data["S_db_images"]
                self.S_db_labels = data["S_db_labels"]
                self.S_db_poses = data["S_db_poses"]
                print("Dataset loaded.")

        # Load from an object
        else:
            self.data_copy(copy_from)

    def print_datasets(self):
        print("Size information for datasets:\n",
              "Training triplets: ", self.train_triplets.shape, "\n",
              "Training images: ", self.S_train_images.shape, "\n",
              "Training poses: ", self.S_train_poses.shape, "\n",
              "Test images: ", self.S_test_images.shape, "\n",
              "Test labels: ", self.S_test_labels.shape, "\n",
              "Test poses: ", self.S_test_poses.shape, "\n",
              "DB images: ", self.S_db_images.shape, "\n",
              "DB labels: ", self.S_db_labels.shape, "\n",
              "DB poses: ", self.S_db_poses.shape, sep="")

    def adjust_channel_axis(self):
        self.train_triplets = np.transpose(self.train_triplets, (0, 3, 1, 2)) # NxHxWxC -> NxCxHxW
        self.S_db_images = np.transpose(self.S_db_images, (0, 3, 1, 2))       # NxHxWxC -> NxCxHxW
        self.S_test_images = np.transpose(self.S_test_images, (0, 3, 1, 2))   # NxHxWxC -> NxCxHxW

    def data_copy(self, obj):
        self.train_triplets = obj.train_triplets
        self.S_train_images = obj.S_train_images
        self.S_train_poses = obj.S_train_poses
        self.S_test_images = obj.S_test_images
        self.S_test_labels = obj.S_test_labels
        self.S_test_poses = obj.S_test_poses
        self.S_db_images = obj.S_db_images
        self.S_db_labels = obj.S_db_labels
        self.S_db_poses = obj.S_db_poses

    def __len__(self):
        if self.type == "train":
            return self.train_triplets.shape[0]
        elif self.type == "test":
            return self.S_test_images.shape[0]
        elif self.type == "db":
            return self.S_db_images.shape[0]
        else:
            raise "ERROR: Unknown dataset type!"

    def __getitem__(self, idx):
        if self.type == "train":
            return self.train_triplets[idx] # idx: 0, 1, 2... [0: anchor0, 1: puller0, 2: pusher0, 3: anchor1, 4: puller1, 5: pusher1...]
        elif self.type == "test":
            return self.S_test_images[idx], self.S_test_labels[idx], self.S_test_poses[idx]
        elif self.type == "db":
            return self.S_db_images[idx], self.S_db_labels[idx], self.S_db_poses[idx]
        else:
            raise "ERROR: Unknown dataset type!"