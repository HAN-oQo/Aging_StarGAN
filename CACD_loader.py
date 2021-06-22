from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import glob
import shutil
import mat73
import numpy as np

estimation = False

class CACD(data.Dataset):
    """Dataset class for the CACD dataset"""
    
    def __init__(self, image_dir, attr_path, age_group, age_group_mode,  additional_dataset, transform, mode ):
        """Initialize and preprocess the CACD dataset / num of data: 163446 / age max: 62, min: 14 """
        
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.age_group_mode = age_group_mode
        if self.age_group_mode == 0: 
            self.age_group = age_group
        elif self.age_group_mode == 1:
            self.age_group = 5
        elif self.age_group_mode == 2:
            self.age_group = 4
        else:
           self.age_group = self.age_group
        self.transform = transform
        self.mode = mode
        self.additional_dataset = additional_dataset
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        
    def preprocess(self):
        """Preprocess the CACD dataset"""

        attr = mat73.loadmat(self.attr_path)
        print("CACD Attribute Loaded")
        image_list = os.listdir(self.image_dir)
        idx = np.arange(len(image_list))
        
        max_age = attr.celebrityImageData.age.max()
        min_age = attr.celebrityImageData.age.min()
        dist_age = ((max_age - min_age)/self.age_group)
 
        random.seed(1234)
        random.shuffle(idx)
        count = 0
        for i in idx:
            count += 1
            #print(i)
            filename = attr.celebrityImageData.name[i][0] #'23_Katie_Findlay_0013.jpg'
            age = attr.celebrityImageData.age[i] # 23.0
            # labels = []
            label = []
            label_age = min_age
            img_age_group = 0
            if self.age_group_mode == 0:
                for j in range(self.age_group):
                    if (age >= min_age + j* (dist_age) and age < min_age + (j+1)*(dist_age)):
                        label.append(1)
                        img_age_group = j

                    else:
                        if max_age ==  (min_age + (j+1)*(dist_age)):
                            if age == max_age:
                                label.append(1)
                                img_age_group = j
                            else:
                                label.append(0)
                        else:
                            label.append(0)
            elif self.age_group_mode == 1: 
                for j in range(self.age_group):
                    if age <= 14:
                        label = [1, 0, 0, 0, 0]
                    elif (age > 14) and (age <= 25):
                        label = [0, 1, 0, 0, 0]
                    elif (age > 25) and (age <= 40):
                        label = [0, 0, 1, 0, 0]
                    elif (age > 40) and (age <= 60):
                        label = [0, 0, 0, 1, 0]
                    elif (age > 60):
                        label = [0, 0, 0, 0, 1]
            
            elif self.age_group_mode == 2: 
                for j in range(self.age_group):
                    age_label = [age - min_age]
                    if (age >= 14) and (age < 26):
                        img_age_group = 0
                        label = [0]
                    elif (age >= 26) and (age < 38):
                        img_age_group = 1
                        label = [1]
                    elif (age >= 38) and (age < 50):
                        img_age_group = 2
                        label = [2]
                    elif (age >= 50) and (age <= 62):
                        img_age_group = 3
                        label = [3]
                    
            else:
                label = [0] * self.age_group
                age_idx = int(age - min_age)
                label[int(age_idx)] = 1
                        
            if self.age_group_mode != 2:           
                if len(label) != self.age_group:
                    print(filename)
                    print(age)
                    print("----------------------error ----------------")
                # labels.append(label)
            
            if not os.path.exists('data/CACD'):
                os.makedirs('data/CACD')
            if not os.path.exists('data/CACD/test'):
                os.makedirs('data/CACD/test')

            src_dir = self.image_dir
            dst_dir = 'data/CACD/test'

            for k in range(self.age_group):
                dir_name = 'age_group{}'.format(k)
                if not os.path.exists(os.path.join(dst_dir, dir_name)):
                    os.makedirs(os.path.join(dst_dir, dir_name))

            if count < 1601:
                if estimation == True:
                    self.test_dataset.append([filename, age_label, label])
                else:
                    jpgfile = os.path.join(src_dir, filename)             
                    self.test_dataset.append([jpgfile, label])             
                
                dst_dir = os.path.join(dst_dir, 'age_group{}'.format(img_age_group))
                if not os.path.exists(os.path.join(dst_dir, filename)):
                    shutil.copy(jpgfile, dst_dir) 

            else:
                jpgfile = os.path.join(src_dir, filename)
                if estimation == True:
                    self.train_dataset.append([jpgfile, age_label, label])
                else: 
                    self.train_dataset.append([jpgfile, label])

        if self.additional_dataset:
            utk_dir = '../UTKFace'
            fgnet_dir = '../FGNET/images'

            utk_list = os.listdir(utk_dir)
            fgnet_dir = os.listdir(fgnet_dir)

            utk_len = len(utk_list)
            fgnet_len = len(fgnet_dir)

            utk_idx = np.arange(utk_len)
            fgnet_idx = np.arange(fgnet_len)

            random.seed(1234)
            random.shuffle(utk_idx)
            random.shuffle(fgnet_idx)

            for i in utk_idx:
                filename = utk_list[i]
                jpgfile = os.path.join(utk_dir, filename)
                age = int(filename.split('_')[0])
                age_label = [age-min_age]
                if (age < 14) and (age > 62):
                    pass

                elif (age >= 14) and (age < 26):
                    label = [0]
                elif (age >= 26) and (age < 38):
                    label = [1]
                elif (age >= 38 ) and (age < 50):
                    label = [2]
                elif (age >= 50) and (age <= 62):
                    label = [3]
                
                
                if estimation == True:
                    self.train_dataset.append([jpgfile, age_label, label])
                else: 
                    self.train_dataset.append([jpgfile, label])
            
            print("UTKFace dataset loaded")


        print("test dataset length: ", len(self.test_dataset))
        print("train dataset length: ", len(self.train_dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label"""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        if estimation ==True:
            jpgfile, age_label, label = dataset[index]
        else:
            jpgfile, label = dataset[index]

        image = Image.open(os.path.join(jpgfile))
        if self.additional_dataset != True:
            return jpgfile, self.transform(image), torch.FloatTensor(label)
        else:
            if image.size == (250, 250):
                if estimation == True:            
                    return jpgfile, self.transform(image), torch.FloatTensor(age_label), torch.FloatTensor(label)
                else: 
                    return jpgfile, self.transform(image), torch.FloatTensor(label)
            else:
                transform0 = []
                transform0.append(T.RandomHorizontalFlip())
                transform0.append(T.Resize(128))
                transform0.append(T.ToTensor())
                transform0.append(T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])) # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5
                transform0 = T.Compose(transform0)
                if estimation == True:
                    return jpgfile, transform0(image), torch.FloatTensor(age_label), torch.FloatTensor(label)
                else:
                    return jpgfile, transform0(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of iamges"""
        return self.num_images



def get_loader2(image_dir, attr_path, age_group, age_group_mode, crop_size = 160, image_size = 128, batch_size = 16, dataset = 'CACD', additional_dataset =True, mode = 'train', num_workers=1):
    """Build and return a data loader"""

    transform = []
    if mode == 'train':
         transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])) # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5
    transform = T.Compose(transform)

    if dataset == 'CACD':
        dataset = CACD(image_dir, attr_path, age_group,age_group_mode, additional_dataset, transform, mode)
    
    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=(mode=='train'),
                                    num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=(mode=='train'),
                                    num_workers=num_workers)

    return data_loader