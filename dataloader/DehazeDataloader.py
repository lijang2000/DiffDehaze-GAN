import argparse
import os
from os import listdir
from os.path import isfile
import torch
import torchvision
import torch.utils.data
import PIL.Image
import re
import random


train_haze_file = "train/haze"
train_clear_file = "train/clear"
test_haze_file = "test/haze"
test_clear_file = "test/clear"
class Dehaze:
    def __init__(self, config):
        self.config = config


        self.transforms = (
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]))

    def get_loaders(self, parse_patches=True, validation='snow'):
        print("=> 加载去雾数据集...")

        
        train_dataset = DehazeDataset(dir=self.config.data.data_dir,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                      parse_patches=parse_patches, flag='train')

        
        val_dataset = DehazeDataset(dir=self.config.data.data_dir,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                    parse_patches=parse_patches, flag='test')

        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)

        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.training.val_batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

class DehazeDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, transforms, filelist=None, parse_patches=True, flag=''):
        if flag == 'train':
            snow100k_dir = dir

            input_names, gt_names = [], []

            haze_inputs = os.path.join(snow100k_dir, train_haze_file)
            gt_inputs = os.path.join(snow100k_dir, train_clear_file)

            images = [f for f in listdir(haze_inputs) if isfile(os.path.join(haze_inputs, f))]
            images2 = [f for f in listdir(gt_inputs) if isfile(os.path.join(gt_inputs, f))]
            print(f"the {flag} datasets len of datasets {len(images)}")

            input_names += [os.path.join(haze_inputs, i) for i in images]

            gt_names += [os.path.join(gt_inputs, i) for i in images2]

            x = list(enumerate(input_names))

            random.shuffle(x)

            indices, input_names = zip(*x)

            gt_names = [gt_names[idx] for idx in indices]

            self.dir = None
            self.flag = 'train'

        else:
            test_dir = dir
            haze_names, gt_names = [], []
            haze_inputs = os.path.join(test_dir, test_haze_file)
            gt_inputs = os.path.join(test_dir, test_clear_file)

            
            haze_image = [f for f in listdir(haze_inputs) if isfile(os.path.join(haze_inputs, f))]
            clear_images = [f for f in listdir(gt_inputs) if isfile(os.path.join(gt_inputs, f))]
            print(f"the {flag} datasets len of datasets {len(haze_image)}")

            
            haze_names += [os.path.join(haze_inputs, i) for i in haze_image]
            
            gt_names += [os.path.join(gt_inputs, i) for i in clear_images]

            x = list(enumerate(haze_names))

            
            random.shuffle(x)

            
            indices, input_names = zip(*x)

            
            gt_names = [gt_names[idx] for idx in indices]

            self.dir = None
            self.flag = 'test'

        self.input_names = input_names
        self.gt_names = gt_names
        self.transforms = transforms
        self.patch_size = patch_size


    def get_image(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split('/',input_name)[-1][:-4]

        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)

        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_image(index)
        return res

    def __len__(self):
        return len(self.input_names)
