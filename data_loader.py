from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import glob
import shutil


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        if not os.path.exists('data/celeba/images/test'):
            os.makedirs('data/celeba/images/test')
        

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
                if not os.path.exists('data/celeba/test'):
                    os.makedirs('data/celeba/test')
                src_dir = 'data/celeba/images'
                dst_dir = 'data/celeba/test'
                jpgfile = os.path.join(src_dir, filename)
                count = 0
                if not os.path.exists(os.path.join(dst_dir, filename)):
                    shutil.copy(jpgfile, dst_dir)
                
            else:
                self.train_dataset.append([filename, label])
        
        ## for counting number of test dataset
        # for path in os.listdir(dst_dir):
        #     if os.path.isfile(os.path.join(dst_dir, path)):
        #         count +=1
        # print(count)

        print(filename)
        print("test dataset length: ", len(self.test_dataset))
        print("train dataset length: ", len(self.train_dataset))


        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return filename, self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    if mode == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=(mode=='train'),
                                    num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=(mode=='train'),
                                    num_workers=num_workers)

    return data_loader