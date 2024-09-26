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
import torchvision.transforms as T
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


class TrainValidateTestDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, mode: str):

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.operation = mode

        # image name without suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # List of files
        self.images_list = [list(self.images_dir.glob(id + '.*'))[0] for id in self.ids]
        self.labels_list = [list(self.labels_dir.glob(id + '.*'))[0] for id in self.ids]

        self.transforms = T.Compose([
            T.ToTensor()
        ])


    def __len__(self):
        if self.operation == "train":
            return len(self.ids) * ph.batch_size
        else:
            return len(self.ids)


    @classmethod
    def label_preprocess(cls, label):
        label[label <= 0.5] = 0
        label[label > 0.5] = 1
        return label

    @classmethod
    def load(cls, filename):
        img = Image.open(filename)
        img = np.array(img).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        idx = idx % len(self.ids)
        img = self.load(self.images_list[idx])
        img = self.transforms(img)

        label = self.load(self.labels_list[idx]).astype(np.float32)
        label /= 255.
        label = self.label_preprocess(label)
        label = torch.from_numpy(label)


        if self.operation == "train":
            x1, x2, y1, y2 = get_random_pos(img, ph.window_size)
            img_cut = img[:, x1:x2, y1:y2]
            label_cut = label[x1:x2, y1:y2]
            return img_cut, label_cut
        else:
            return img, label

