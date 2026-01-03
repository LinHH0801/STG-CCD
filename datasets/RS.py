import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
from torchvision import transforms as T
# from osgeo import gdal_array
import cv2

num_classes = 2
ST_COLORMAP = [[0, 0,0], [255, 255, 255]]
ST_CLASSES = ['unchanged', 'change']

MEAN = np.array([0.5, 0.5, 0.5])
STD = np.array([0.5, 0.5, 0.5])


root = r'D:\LHH_BSYJ\Datasets\OpenWUSU512'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im):
    im = (im - MEAN) / STD
    return im


def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def read_RSimages(mode, rescale=False):
    # assert mode in ['train', 'val', 'test']
    img_1_dir = os.path.join(root, mode, 'im15_rgb')
    img_2_dir = os.path.join(root, mode, 'im16_rgb')
    img_3_dir = os.path.join(root, mode, 'im18_rgb')
    label_T12_dir = os.path.join(root, mode, 'label1516_hsv')
    label_T23_dir = os.path.join(root, mode, 'label1618_hsv')

    label_A_dir = os.path.join(root, mode, 'labelBCD_hsv')

    data_list = os.listdir(img_1_dir)

    imgs_list_1 = []
    imgs_list_2 = []
    imgs_list_3 = []
    labels_12 = []
    labels_23 = []
    labels = []
    count = 0
    for it in data_list:

        if (it[-4:] == '.tif'):
            img_1_path = os.path.join(img_1_dir, it)
            img_2_path = os.path.join(img_2_dir, it)
            img_3_path = os.path.join(img_3_dir, it)
            label_12_path = os.path.join(label_T12_dir, it)
            label_23_path = os.path.join(label_T23_dir, it)
            label_path = os.path.join(label_A_dir, it)

            imgs_list_1.append(img_1_path)
            imgs_list_2.append(img_2_path)
            imgs_list_3.append(img_3_path)
            label12 = io.imread(label_12_path)
            label23 = io.imread(label_23_path)
            label = io.imread(label_path)
            labels_12.append(label12)
            labels_23.append(label23)
            labels.append(label)

        count += 1
        if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))

    print(labels[0].shape)
    print(str(len(imgs_list_1)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_1, imgs_list_2, imgs_list_3, labels_12,labels_23,labels

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_1, self.imgs_list_2,  self.imgs_list_3,  self.labels_12,self.labels_23,self.labels = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_1[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_1 = normalize_image(io.imread(self.imgs_list_1[idx]))
        img_2 = normalize_image(io.imread(self.imgs_list_2[idx]))
        img_3 = normalize_image(io.imread(self.imgs_list_3[idx]))
        label12 = self.labels_12[idx]
        label23 = self.labels_23[idx]
        label = self.labels[idx]
        if self.random_flip:
            img_1, img_2,img_3,label12,label23,label = transform.rand_rot90_flip_MCD(img_1, img_2,img_3, label12,label23,label)
        return F.to_tensor(img_1), F.to_tensor(img_2),F.to_tensor(img_3),torch.from_numpy(label12),torch.from_numpy(label23),torch.from_numpy(label)
    def __len__(self):
        return len(self.imgs_list_1)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_1 = []
        self.imgs_2 = []
        self.imgs_3 = []
        self.mask_name_list = []
        img_1_dir = os.path.join(test_dir, 'im15_rgb')
        img_2_dir = os.path.join(test_dir, 'im16_rgb')
        img_3_dir = os.path.join(test_dir, 'im18_rgb')
        data_1_list = os.listdir(img_1_dir)

        for it in data_1_list:
            if (it[-4:] == '.tif'):
                img_1_path = os.path.join(img_1_dir, it)
                img_2_path = os.path.join(img_2_dir, it)
                img_3_path = os.path.join(img_3_dir, it)

                self.imgs_1.append(io.imread(img_1_path))
                self.imgs_2.append(io.imread(img_2_path))
                self.imgs_3.append(io.imread(img_3_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_1)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_1 = normalize_image(self.imgs_1[idx])
        img_2 = normalize_image(self.imgs_2[idx])
        img_3 = normalize_image(self.imgs_3[idx])
        # img_4 = normalize_image(self.imgs_4[idx])

        return F.to_tensor(img_1), F.to_tensor(img_2),F.to_tensor(img_3)

    def __len__(self):
        return self.len



