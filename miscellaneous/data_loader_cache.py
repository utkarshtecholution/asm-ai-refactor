from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
import json
from tqdm import tqdm
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

# Acknowledgement:
# We would like to thank Dr. Ibrahim Almakky (https://scholar.google.co.uk/citations?user=T9MTcK0AAAAJ&hl=en)
# for his helps in implementing cache mechanism of our DIS dataloader.

def get_im_gt_name_dict(datasets, flag='valid'):
    """
    Create a list of dictionaries containing image and ground truth paths for given datasets.

    Parameters:
        datasets (list): List of dataset configurations.
        flag (str): Dataset flag, either 'train' or 'valid'.

    Returns:
        list: List of dictionaries with dataset information.
    """
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i, dataset in enumerate(datasets):
        print(f"--->>> {flag} dataset {i}/{len(datasets)} {dataset['name']} <<<---")
        tmp_im_list = glob(os.path.join(dataset["im_dir"], '*' + dataset["im_ext"]))
        print(f'-im- {dataset["name"]} {dataset["im_dir"]}: {len(tmp_im_list)}')

        if dataset["gt_dir"] == "":
            print(f'-gt- {dataset["name"]} {dataset["gt_dir"]}: No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [os.path.join(dataset["gt_dir"], os.path.basename(x).split(dataset["im_ext"])[0] + dataset["gt_ext"]) for x in tmp_im_list]
            print(f'-gt- {dataset["name"]} {dataset["gt_dir"]}: {len(tmp_gt_list)}')

        if flag == "train":  # Combine multiple training sets into one dataset
            if not name_im_gt_list:
                name_im_gt_list.append({
                    "dataset_name": dataset["name"],
                    "im_path": tmp_im_list,
                    "gt_path": tmp_gt_list,
                    "im_ext": dataset["im_ext"],
                    "gt_ext": dataset["gt_ext"],
                    "cache_dir": dataset["cache_dir"]
                })
            else:
                name_im_gt_list[0]["dataset_name"] += f"_{dataset['name']}"
                name_im_gt_list[0]["im_path"] += tmp_im_list
                name_im_gt_list[0]["gt_path"] += tmp_gt_list
                if dataset["im_ext"] != ".jpg" or dataset["gt_ext"] != ".png":
                    raise ValueError("Error: Please make sure all your images and ground truth masks are in jpg and png format respectively!")
                name_im_gt_list[0]["im_ext"] = ".jpg"
                name_im_gt_list[0]["gt_ext"] = ".png"
                name_im_gt_list[0]["cache_dir"] = os.path.join(os.path.dirname(dataset["cache_dir"]), name_im_gt_list[0]["dataset_name"])
        else:  # Keep different validation or inference datasets as separate ones
            name_im_gt_list.append({
                "dataset_name": dataset["name"],
                "im_path": tmp_im_list,
                "gt_path": tmp_gt_list,
                "im_ext": dataset["im_ext"],
                "gt_ext": dataset["gt_ext"],
                "cache_dir": dataset["cache_dir"]
            })

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, cache_size=[], cache_boost=True, my_transforms=[], batch_size=1, shuffle=False):
    """
    Create dataloaders for the given image and ground truth paths.

    Parameters:
        name_im_gt_list (list): List of dictionaries with dataset information.
        cache_size (list): Size to which images and ground truths will be resized.
        cache_boost (bool): Flag to enable or disable cache boost.
        my_transforms (list): List of transformations to be applied to the dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Flag to shuffle the dataset.

    Returns:
        tuple: List of DataLoader objects and list of GOSDatasetCache objects.
    """
    gos_dataloaders = []
    gos_datasets = []

    if not name_im_gt_list:
        return gos_dataloaders, gos_datasets

    num_workers = min(8, max(1, batch_size // 4))

    for name_im_gt in name_im_gt_list:
        gos_dataset = GOSDatasetCache(
            [name_im_gt],
            cache_size=cache_size,
            cache_path=name_im_gt["cache_dir"],
            cache_boost=cache_boost,
            transform=transforms.Compose(my_transforms)
        )
        gos_dataloaders.append(DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

def im_reader(im_path):
    """
    Read an image from the given path.

    Parameters:
        im_path (str): Path to the image file.

    Returns:
        np.array: Image array.
    """
    return io.imread(im_path)

def im_preprocess(im, size):
    """
    Preprocess an image by resizing and normalizing.

    Parameters:
        im (np.array): Image array.
        size (list): Size to which the image will be resized.

    Returns:
        tuple: Preprocessed image tensor and its original shape.
    """
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=-1)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)

    im_tensor = torch.tensor(im.copy(), dtype=torch.float32).permute(2, 0, 1)
    if len(size) < 2:
        return im_tensor, im.shape[:2]
    im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

    return im_tensor.to(torch.uint8), im.shape[:2]

def gt_preprocess(gt, size):
    """
    Preprocess a ground truth mask by resizing and normalizing.

    Parameters:
        gt (np.array): Ground truth array.
        size (list): Size to which the ground truth will be resized.

    Returns:
        tuple: Preprocessed ground truth tensor and its original shape.
    """
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    gt_tensor = torch.tensor(gt, dtype=torch.uint8).unsqueeze(0)
    if len(size) < 2:
        return gt_tensor, gt.shape[:2]
    gt_tensor = F.interpolate(gt_tensor.unsqueeze(0).float(), size=size, mode="bilinear", align_corners=False).squeeze(0)

    return gt_tensor.to(torch.uint8), gt.shape[:2]

class GOSRandomHFlip(object):
    """
    Random horizontal flip transformation.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample['image'] = torch.flip(sample['image'], dims=[2])
            sample['label'] = torch.flip(sample['label'], dims=[2])
        return sample

class GOSResize(object):
    """
    Resize transformation.
    """
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        sample['label'] = F.interpolate(sample['label'].unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        return sample

class GOSRandomCrop(object):
    """
    Random crop transformation.
    """
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[1:]
        new_h, new_w = self.size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        sample['image'] = image[:, top:top + new_h, left:left + new_w]
        sample['label'] = label[:, top:top + new_h, left:left + new_w]
        return sample

class GOSNormalize(object):
    """
    Normalize transformation.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = normalize(sample['image'], self.mean, self.std)
        return sample

class GOSDatasetCache(Dataset):
    """
    Custom Dataset class with caching mechanism for loading images and ground truth masks.
    """
    def __init__(self, name_im_gt_list, cache_size=[], cache_path='./cache', cache_file_name='dataset.json', cache_boost=False, transform=None):
        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        self.cache_boost = cache_boost
        self.transform = transform
        self.dataset = self.combine_datasets(name_im_gt_list)
        self.dataset = self.manage_cache()

    def combine_datasets(self, name_im_gt_list):
        """
        Combine different datasets into one.

        Parameters:
            name_im_gt_list (list): List of dictionaries with dataset information.

        Returns:
            dict: Combined dataset dictionary.
        """
        dataset = {
            "data_name": [],
            "im_name": [],
            "im_path": [],
            "ori_im_path": [],
            "gt_path": [],
            "ori_gt_path": [],
            "im_shp": [],
            "gt_shp": [],
            "im_ext": [],
            "gt_ext": [],
            "ims_pt_dir": "",
            "gts_pt_dir": ""
        }

        for item in name_im_gt_list:
            dataset["data_name"].extend([item["dataset_name"]] * len(item["im_path"]))
            dataset["im_name"].extend([os.path.basename(x).split(item["im_ext"])[0] for x in item["im_path"]])
            dataset["im_path"].extend(item["im_path"])
            dataset["ori_im_path"].extend(deepcopy(item["im_path"]))
            dataset["gt_path"].extend(item["gt_path"])
            dataset["ori_gt_path"].extend(deepcopy(item["gt_path"]))
            dataset["im_ext"].extend([item["im_ext"]] * len(item["im_path"]))
            dataset["gt_ext"].extend([item["gt_ext"]] * len(item["gt_path"]))

        return dataset

    def manage_cache(self):
        """
        Manage the caching of the dataset.

        Returns:
            dict: Cached dataset dictionary.
        """
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        dataset_names = "_".join(set(self.dataset["data_name"]))
        cache_folder = os.path.join(self.cache_path, f"{dataset_names}_{'x'.join(map(str, self.cache_size))}")

        if not os.path.exists(cache_folder):
            return self.cache_data(cache_folder)
        return self.load_cache(cache_folder)

    def cache_data(self, cache_folder):
        """
        Cache the dataset by saving images and ground truths to files.

        Parameters:
            cache_folder (str): Path to the cache folder.

        Returns:
            dict: Cached dataset dictionary.
        """
        os.makedirs(cache_folder)
        cached_dataset = deepcopy(self.dataset)
        ims_pt_list = []
        gts_pt_list = []

        for i, im_path in tqdm(enumerate(self.dataset["im_path"]), total=len(self.dataset["im_path"])):
            im_id = cached_dataset["im_name"][i]
            im = im_reader(im_path)
            im, im_shp = im_preprocess(im, self.cache_size)
            im_cache_file = os.path.join(cache_folder, f"{self.dataset['data_name'][i]}_{im_id}_im.pt")
            torch.save(im, im_cache_file)
            cached_dataset["im_path"][i] = im_cache_file

            if self.cache_boost:
                ims_pt_list.append(im.unsqueeze(0))

            gt = np.zeros(im.shape[:2])
            if self.dataset["gt_path"][i]:
                gt = im_reader(self.dataset["gt_path"][i])
            gt, gt_shp = gt_preprocess(gt, self.cache_size)
            gt_cache_file = os.path.join(cache_folder, f"{self.dataset['data_name'][i]}_{im_id}_gt.pt")
            torch.save(gt, gt_cache_file)
            cached_dataset["gt_path"][i] = gt_cache_file

            if self.cache_boost:
                gts_pt_list.append(gt.unsqueeze(0))

            cached_dataset["im_shp"].append(im_shp)
            cached_dataset["gt_shp"].append(gt_shp)

        if self.cache_boost:
            cached_dataset["ims_pt_dir"] = os.path.join(cache_folder, f"{self.cache_file_name.split('.json')[0]}_ims.pt")
            cached_dataset["gts_pt_dir"] = os.path.join(cache_folder, f"{self.cache_file_name.split('.json')[0]}_gts.pt")
            torch.save(torch.cat(ims_pt_list, dim=0), cached_dataset["ims_pt_dir"])
            torch.save(torch.cat(gts_pt_list, dim=0), cached_dataset["gts_pt_dir"])

        with open(os.path.join(cache_folder, self.cache_file_name), "w") as json_file:
            json.dump(cached_dataset, json_file)

        return cached_dataset

    def load_cache(self, cache_folder):
        """
        Load the cached dataset from files.

        Parameters:
            cache_folder (str): Path to the cache folder.

        Returns:
            dict: Loaded cached dataset dictionary.
        """
        with open(os.path.join(cache_folder, self.cache_file_name), "r") as json_file:
            dataset = json.load(json_file)

        if self.cache_boost:
            self.ims_pt = torch.load(dataset["ims_pt_dir"], map_location='cpu')
            self.gts_pt = torch.load(dataset["gts_pt_dir"], map_location='cpu')

        return dataset

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        if self.cache_boost and self.ims_pt is not None:
            im = self.ims_pt[idx]
            gt = self.gts_pt[idx]
        else:
            im = torch.load(self.dataset["im_path"][idx])
            gt = torch.load(self.dataset["gt_path"][idx])

        im_shp = self.dataset["im_shp"][idx]
        im = im.float() / 255.0
        gt = gt.float() / 255.0

        sample = {
            "imidx": torch.tensor(idx),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im_shp)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample