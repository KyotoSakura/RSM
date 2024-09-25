import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.path_hyperparameter import ph
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class BasicDataset(Dataset):
    """ Basic dataset for train, evaluation and test.
    
    Attributes:
        images_dir(str): path of images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        ids(list): name list of images.
        train_transforms_all(class): data augmentation applied to image and label.

    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        """ Init of basic dataset.
        
        Parameter:
            images_dir(str): path of images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.

        """
        
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # List of files
        self.images_list = [list(self.images_dir.glob(id + '.*'))[0] for id in self.ids]
        self.labels_list = [list(self.labels_dir.glob(id + '.*'))[0] for id in self.ids]

        self.train_transforms_all = A.Compose([
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # 使用最简单的数据增强方法
            # A.Rotate(45, p=0.3),
            # A.ShiftScaleRotate(p=0.3),
        ], additional_targets={'image1': 'image'})

        self.normalize = A.Compose([
            A.Normalize()
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ])

    def __len__(self):
        """ Return length of dataset."""
        return len(self.ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label[label != 0] = 1
        return label

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        img = np.array(img).astype(np.uint8)

        return img
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, idx):
        """ Index dataset.

        Index image name list to get image name, search image in image path with its name,
        open image and convert it to array.

        Preprocess array, apply data augmentation and noise addition(optional) on it, and convert array to tensor.

        Parameter:
            idx(int): index of dataset.

        Return:
            tensor(tensor): tensor of image.
            label_tensor(tensor): tensor of label.
            name(str): the same name of image and label.
        """

        
        # name = self.ids[idx]
        # img_file = list(self.images_dir.glob(name + '.*'))
        # label_file = list(self.labels_dir.glob(name + '.*'))

        # assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {name}: {label_file}'
        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # Pick a random image and its label
        random_idx = random.randint(0, len(self.images_list) - 1)
        
        # Convert to array
        img = self.load(self.images_list[random_idx]).astype(np.float32)
        img = img.transpose(2, 0, 1)
        label = self.load(self.labels_list[random_idx]).astype(np.float32)
        label = self.label_preprocess(label)
        
        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(img, ph.window_size)
        img_p = img[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        # if self.train:
        #     sample = self.train_transforms_all(image=img_p, mask=label_p)
        #     img, label = sample['image'], sample['mask']
        img_p, label_p = self.data_augmentation(img_p, label_p)

        # img_p = img_p / 255.
        # label_p = label_p / 255. 
        # Convert to tensor
        
        # img = self.normalize(image=img)['image']
        # img_label_assemble_tensor = self.to_tensor(image=img, mask=label)
        # ipdb.set_trace()
        # img_tensor, label_tensor = img_label_assemble_tensor['image'].contiguous(), img_label_assemble_tensor['mask'].contiguous()

        return (torch.from_numpy(img_p), torch.from_numpy(label_p))
