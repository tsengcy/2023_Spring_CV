import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(dataset_dir, batch_size=1, split='test'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform2 = transforms.Compose([
        transforms.Resize((32,32)),
        ##### TODO: Data Augmentation Begin #####
        ##### TODO: Data Augmentation End #####
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform, transform2=transform2)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None, transform2 = None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        self.transform2 = transform2

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        
        # some modify

        # self.flag = [0] * len(json_data['filenames'])
        

        if self.split != 'test':
            self.labels = json_data['labels']
            # some modify
            
        
        # if(self.split == 'train'):
        #     self.image_names.extend(json_data['filenames'])
        #     self.labels.extend(json_data['labels'])
        #     self.flag.extend([1] * len(json_data['filenames']))

        

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################

        img = Image.open(self.dataset_dir + '/' + self.image_names[index]).convert('RGB')
        
        # if self.flag == 1:
        #     img = self.transform2(img)
        # else:
        #     if self.transform:
        #         img = self.transform(img)
        if self.transform:
            img = self.transform(img)

        if self.split == 'test':
            return {
                'images': img
            }
        else:
            return {
                'images': img, 
                'labels': self.labels[index]
            }
        
        # return {
        #     'images': img, 
        #     'labels': self.labels[index]
        # }
