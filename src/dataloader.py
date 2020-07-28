import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np

class VisionDataset(object):
    """
    Code to load the dataloaders for the storage memory (implicitly performs greedy sampling) for training GDumb. Should be easily readable and extendable to any new dataset.
    Should generate class_mask, cltrain_loader, cltest_loader; with support for pretraining dataloaders given as pretrain_loader and pretest_loader.
    """
    def __init__(self, opt, class_order=None):
        self.kwargs = {
        'num_workers': opt.workers,
        'batch_size': opt.batch_size,
        'shuffle': True,
        'pin_memory': True}
        self.opt = opt
        # Sets parameters of the dataset. For adding new datasets, please add the dataset details in `get_statistics` function.
        mean, std, opt.total_num_classes, opt.inp_size, opt.in_channels = get_statistics(opt.dataset)
        self.class_order = class_order
        # Generates the standard data augmentation transforms
        train_augment, test_augment = get_augment_transforms(dataset=opt.dataset, inp_sz=opt.inp_size)
        self.train_transforms = torchvision.transforms.Compose(train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        self.test_transforms = torchvision.transforms.Compose(test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

        # Creates the supervised baseline dataloader (upper bound for continual learning methods)
        self.supervised_trainloader = self.get_loader(indices=None, transforms=self.train_transforms, train=True)
        self.supervised_testloader = self.get_loader(indices=None, transforms=self.test_transforms, train=False)
        self.kwargs['shuffle'] = False

    def get_loader(self, indices, transforms, train, shuffle=True, target_transforms=None):
        sampler = None
        if indices is not None: sampler = SubsetRandomSampler(indices) if (shuffle and train) else SubsetSequentialSampler(indices)       
        
        # Support for *some* pytorch default loaders is provided. Code is made such that adding new datasets is super easy, given they are in ImageFolder format.        
        if self.opt.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'KMNIST', 'FashionMNIST']:
            return DataLoader(getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, train=train, download=True, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)
        elif self.opt.dataset=='SVHN':
            split = 'train' if train else 'test'
            return DataLoader(getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, split=split, download=True, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)       
        else:
            subfolder = 'train' if train else 'test' # ImageNet 'val' is labled as 'test' here.
            return DataLoader(torchvision.datasets.ImageFolder(self.opt.data_dir+'/'+self.opt.dataset+'/'+subfolder, transform=transforms, target_transform=target_transforms), sampler=sampler, **self.kwargs)
            
    def gen_cl_mapping(self):
        # Get the label -> idx mapping dictionary
        if self.opt.dataset=='SVHN':
            train_class_labels_dict, test_class_labels_dict = classwise_split(targets=self.supervised_trainloader.dataset.labels), classwise_split(targets=self.supervised_testloader.dataset.labels)
        else:
            train_class_labels_dict, test_class_labels_dict = classwise_split(targets=self.supervised_trainloader.dataset.targets), classwise_split(targets=self.supervised_testloader.dataset.targets)

        # Sets classes to be 0 to n-1 if class order is not specified, else sets it to class order. To produce different effects tweak here.
        class_list = self.class_order if self.class_order is not None else list(range(self.opt.total_num_classes))
        assert(self.opt.num_tasks*self.opt.num_classes_per_task <= self.opt.total_num_classes), "num_classes lesser than classes_per_task * num_tasks"
        pretrain_class_list = class_list[:self.opt.num_pretrain_classes]
        cl_class_list = class_list[:self.opt.num_classes_per_task*self.opt.num_tasks]
        if self.class_order is None: random.shuffle(cl_class_list) # Generates different class-to-task assignment
        
        if self.opt.num_pretrain_classes > 0:
            pretrain_target_transform = ReorderTargets(pretrain_class_list) # Uses target_transforms to remap the class order according to class list
            assert(len(pretrain_class_list)==self.opt.num_pretrain_classes), "Error in generating the pretraining list"
            pretrainidx, pretestidx = [], []
            for cl in pretrain_class_list: # Selects classes from the pretraining list and loads all indices, which are then passed to a subset sampler
                pretrainidx += train_class_labels_dict[cl][:]
                pretestidx += test_class_labels_dict[cl][:]
            self.pretrain_loader = self.get_loader(indices=pretrainidx, transforms=self.train_transforms, train=True, target_transforms=pretrain_target_transform)
            self.pretest_loader = self.get_loader(indices=pretestidx, transforms=self.test_transforms, train=False, target_transforms=pretrain_target_transform)
        
        self.class_mask = torch.from_numpy(np.kron(np.eye(self.opt.num_tasks,dtype=int),np.ones((self.opt.num_classes_per_task,self.opt.num_classes_per_task)))).cuda() #Generates equal num_classes for all tasks. 
        continual_target_transform = ReorderTargets(cl_class_list)  # Remaps the class order to a 0-n order, required for crossentropy loss using class list
        trainidx, testidx = [], []
        mem_per_cls = self.opt.memory_size//(self.opt.num_classes_per_task*self.opt.num_tasks)
        for cl in cl_class_list: # Selects classes from the continual learning list and loads memory and test indices, which are then passed to a subset sampler
            num_memory_samples = min(len(train_class_labels_dict[cl][:]), mem_per_cls)
            trainidx += train_class_labels_dict[cl][:num_memory_samples] # This is class-balanced greedy sampling (Selects the first n samples).
            testidx += test_class_labels_dict[cl][:]
        assert(len(trainidx) <= self.opt.memory_size), "ERROR: Cannot exceed max. memory samples!"
        self.cltrain_loader = self.get_loader(indices=trainidx, transforms=self.train_transforms, train=True, target_transforms=continual_target_transform)
        self.cltest_loader = self.get_loader(indices=testidx, transforms=self.test_transforms, train=False, target_transforms=continual_target_transform)


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)



class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order) 

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]


def get_augment_transforms(dataset, inp_sz):
    """
    Returns appropriate augmentation given dataset size and name
    Arguments:
        indices (sequence): a sequence of indices
    """
    if inp_sz == 32 or inp_sz == 28 or inp_sz == 64:
       train_augment = [torchvision.transforms.RandomCrop(inp_sz, padding=4)]
       test_augment = []
    else:
       train_augment = [torchvision.transforms.RandomResizedCrop(inp_sz)]
       test_augment = [torchvision.transforms.Resize(inp_sz+32), torchvision.transforms.CenterCrop(inp_sz)] 
    
    if dataset not in ['MNIST', 'SVHN', 'KMNIST']:
        train_augment.append(torchvision.transforms.RandomHorizontalFlip()) 

    return train_augment, test_augment

        
def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict

def get_statistics(dataset):
    '''
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    '''
    assert(dataset in ['MNIST', 'KMNIST', 'EMNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'CINIC10', 'ImageNet100', 'ImageNet', 'TinyImagenet'])
    mean = {
            'MNIST':(0.1307,),
            'KMNIST':(0.1307,),
            'EMNIST':(0.1307,),
            'FashionMNIST':(0.1307,),
            'SVHN':  (0.4377,  0.4438,  0.4728),
            'CIFAR10':(0.4914, 0.4822, 0.4465),
            'CIFAR100':(0.5071, 0.4867, 0.4408),
            'CINIC10':(0.47889522, 0.47227842, 0.43047404),
            'TinyImagenet':(0.4802, 0.4481, 0.3975),
            'ImageNet100':(0.485, 0.456, 0.406),
            'ImageNet':(0.485, 0.456, 0.406),
        }

    std = {
            'MNIST':(0.3081,),
            'KMNIST':(0.3081,),
            'EMNIST':(0.3081,),
            'FashionMNIST':(0.3081,),
            'SVHN': (0.1969,  0.1999,  0.1958),
            'CIFAR10':(0.2023, 0.1994, 0.2010),
            'CIFAR100':(0.2675, 0.2565, 0.2761),
            'CINIC10':(0.24205776, 0.23828046, 0.25874835),
            'TinyImagenet':(0.2302, 0.2265, 0.2262),
            'ImageNet100':(0.229, 0.224, 0.225),
            'ImageNet':(0.229, 0.224, 0.225),
        }

    classes = {
            'MNIST': 10,
            'KMNIST': 10,
            'EMNIST': 49,
            'FashionMNIST': 10,
            'SVHN': 10,
            'CIFAR10': 10,
            'CIFAR100': 100,
            'CINIC10': 10,
            'TinyImagenet':200,
            'ImageNet100':100,
            'ImageNet': 1000,
        }

    in_channels = {
            'MNIST': 1,
            'KMNIST': 1,
            'EMNIST': 1,
            'FashionMNIST': 1,
            'SVHN': 3,
            'CIFAR10': 3,
            'CIFAR100': 3,
            'CINIC10': 3,
            'TinyImagenet':3,
            'ImageNet100':3,
            'ImageNet': 3,
        }

    inp_size = {
            'MNIST': 28,
            'KMNIST': 28,
            'EMNIST': 28,
            'FashionMNIST': 28,
            'SVHN': 32,
            'CIFAR10': 32,
            'CIFAR100': 32,
            'CINIC10': 32,
            'TinyImagenet':64,
            'ImageNet100':224,
            'ImageNet': 224,
        }
    return mean[dataset], std[dataset], classes[dataset],  inp_size[dataset], in_channels[dataset]
