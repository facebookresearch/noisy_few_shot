import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from learn2learn.vision.datasets import TieredImagenet, CIFARFS

from mini_imagenet import MiniImagenet, MiniImagenetOutlier


IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STDS = (0.229, 0.224, 0.225)

CIFAR_MEANS = (0.5071, 0.4867, 0.4408)
CIFAR_STDS = (0.2675, 0.2565, 0.2761)

IMAGENET_SIZE = (84,84)
CIFAR_SIZE = (32,32)


def get_augmentation_transforms(args):
    transform_list = []
    
    if args.dataset == 'miniimagenet':
        transform_list.append(transforms.ToPILImage())
    
    if args.random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if args.random_resized_crop:
        if "imagenet" in args.dataset:
            transform_list.append(transforms.RandomResizedCrop(size=IMAGENET_SIZE, scale=(0.8,1.0)))
        elif "cifar" in args.dataset:
            transform_list.append(transforms.RandomResizedCrop(size=CIFAR_SIZE, scale=(0.8,1.0)))
        else:
            raise NotImplementedError
        
    if args.color_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

    transform_list.append(transforms.ToTensor())
    
    if "imagenet" in args.dataset:
        transform_list.append(transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS))
    elif "cifar" in args.dataset:
        transform_list.append(transforms.Normalize(CIFAR_MEANS, CIFAR_STDS))
    else:
        raise NotImplementedError
        
    if args.random_erasing:
        transform_list.append(transforms.RandomErasing(value="random"))
        
    return transforms.Compose(transform_list)

def get_data_loaders(args):
    if "imagenet" in args.dataset:
        normalize_transform = transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    elif "cifar" in args.dataset:
        normalize_transform = transforms.Normalize(CIFAR_MEANS, CIFAR_STDS)
    else:
        raise NotImplementedError
    
    if args.dataset == 'miniimagenet':
        train_dataset = MiniImagenet(
            root=args.data_path, mode='train', 
            transform=get_augmentation_transforms(args)
        )
        valid_dataset = MiniImagenet(
            root=args.data_path, mode='validation',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
        test_dataset = MiniImagenet(
            root=args.data_path, mode='test',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
    elif args.dataset == 'tieredimagenet':
        train_dataset = TieredImagenet(
            root=args.data_path, mode='train', 
            transform=get_augmentation_transforms(args)
        )
        valid_dataset = TieredImagenet(
            root=args.data_path, mode='validation',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
        test_dataset = TieredImagenet(
            root=args.data_path, mode='test',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )    
    elif args.dataset == 'cifarfs':
        train_dataset = CIFARFS(
            root=args.data_path, mode='train', 
            transform=get_augmentation_transforms(args)
        )
        valid_dataset = CIFARFS(
            root=args.data_path, mode='validation',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
        test_dataset = CIFARFS(
            root=args.data_path, mode='test',
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
    else:
        raise NotImplementedError
        
    if args.noise_type == 'outlier' and "imagenet" in args.dataset:
        outlier_train_dataset = MiniImagenetOutlier(
            root=os.path.join(args.data_path, "../miniimagenet_outlier"), mode="train",
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
        outlier_test_dataset = MiniImagenetOutlier(
            root=os.path.join(args.data_path, "../miniimagenet_outlier"), mode="test",
            transform=transforms.Compose([transforms.ToTensor(), normalize_transform])
        )
        

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + 2*args.train_shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + 2*args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + 2*args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=args.test_tasks)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)
    
    if args.noise_type == 'outlier':
        outlier_train_loader = DataLoader(
            outlier_train_dataset, batch_size=(args.train_way * args.train_shot), pin_memory=True, shuffle=True
        )
        outlier_test_loader = DataLoader(
            outlier_test_dataset, batch_size=(args.test_way * args.test_shot), pin_memory=True, shuffle=True
        )
    else:
        outlier_train_loader = None
        outlier_test_loader = None
        
    return {
        "train": train_loader, 
        "valid": valid_loader, 
        "test": test_loader, 
        "outlier_train": outlier_train_loader,
        "outlier_test": outlier_test_loader,
    }


##################### 
# Data Pre-processing
#####################

def preprocess_data_labels(batch, device):
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    return data, labels


def get_support_noise_query_indices(ways, shot, query_num):
    # Init support and noise arrays
    support_indices = np.zeros(ways * (2*shot + query_num), dtype=bool)
    noise_indices = support_indices.copy()
    
    # Marker for beginning of each of the ways
    selection = np.arange(ways) * (2*shot + query_num)
    
    # Mark support indices, starting from the beginning of each class
    for offset in range(shot):
        support_indices[selection + offset] = True
        noise_indices[selection + shot + offset] = True
    # Query indices are those that aren't support or noise
    query_indices = ~(support_indices | noise_indices)

    # Convert to torch
    return {
        "support": torch.from_numpy(support_indices),
        "noise": torch.from_numpy(noise_indices),
        "query": torch.from_numpy(query_indices),
    }