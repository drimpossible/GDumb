import random
import torch
import numpy as np
import os
import logging
from torch import nn

class AverageMeter:
    # Sourced from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum*1.0 / self.count*1.0

def get_logger(folder):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fh = logging.FileHandler(os.path.join(folder, 'checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_accuracy(y_prob, y_true, class_mask, return_vec=False):
    '''
    Calculates the task and class incremental accuracy of the model
    '''
    y_pred = torch.argmax(y_prob, axis=1)

    mask = class_mask[y_true]
    #assert (y_prob.size() == mask.size()), "Class mask does not match probabilities in output"
    masked_prob = torch.mul(y_prob, mask)
    y_pred_masked = torch.argmax(masked_prob, axis=1)

    acc_full = torch.eq(y_pred, y_true)
    acc_masked = torch.eq(y_pred_masked, y_true)
    if return_vec:
        return acc_full, acc_masked

    return (acc_full*1.0).mean(), (acc_masked*1.0).mean()


def seed_everything(seed):
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # An exemption for speed :P


def save_model(opt, model):
    '''
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    '''
    state = {'opt': opt,
        'state_dict': model.state_dict()}
    filename = opt.log_dir+opt.old_exp_name+'/pretrained_model.pth.tar'
    torch.save(state, filename)


def load_model(opt, model, logger):
    '''
    Used for loading the pretrained model, not for intermediate breaks in running the code.
    '''
    filepath = opt.log_dir+opt.old_exp_name+'/pretrained_model.pth.tar'
    assert(os.path.isfile(filepath))
    logger.debug("=> loading checkpoint '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
