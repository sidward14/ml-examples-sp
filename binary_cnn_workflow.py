"""Boilerplate Binary Classifier CNN code from start to finish."""

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
# import albumentations as A

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = '../data/ml'
ROOT_TEST = '../data/ml/test'
IMG_DIR_SIGS = 'imgs'
IMG_DIR_NOISE = 'imgs_noise'

DEVICE = 'cuda:0'

VALID_PCT = .05
K = None

TRAIN_DL_PARAMS = {
    batch_size = 16,
    shuffle = True,
    drop_last = False,
    ___
}

VALID_DL_PARAMS = {
    batch_size = 16,
    shuffle = True,
    drop_last = False,
    ___
}

TEST_DL_PARAMS = {
    batch_size = 16,
    shuffle = True,
    drop_last = False,
    ___
}

WHICH_MODEL = 'efficientnet-b2'

NUM_ITERS_VALID = 500
NUM_ITERS_LOSS = 500
# NUM_ITERS_SAVE = 500

LR = 5.e-5
NUM_EPOCHS = 10
WHICH_LOSS = 'bce'  # 'focal'
WHICH_OPTIM = 'adam'  # 'sgd'
USE_SCHED = True


class ScratchArcDataset(Dataset):
    def __init__(self, img_paths, gts):
        self.img_paths = img_paths
        self.gts = gts

        self._len = len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.BGR2RGB(img)
        img /= 255.
        
        gt = self.gts[idx]

        return img, gt

    def __len__(self):
        return self._len
 

def get_data(root):
    _img_dir = os.path.join(root, IMG_DIR_SIGS)
    img_paths = [os.path.join(_img_dir, p) for p in os.path.listdir(_img_dir) if p.endswith('.png')]
    gts = [1]*len(img_paths)
    _img_dir = os.path.join(root, IMG_DIR_NOISE)
    _imgs_noise = [os.path.join(_img_dir, p) for p in os.path.listdir(_img_dir) if p.endswith('.png')]
    img_paths += _imgs_noise
    gts += [0]*len(_imgs_noise)
    print(len([i for i in gts if i == 1]))
    print(len([i for i in gts if i == 0]))
    gts = torch.tensor(gts)
    print((gts == 1).sum())
    print((gts == 0).sum())
    assert (len(img_paths) == len(gts))

    return img_paths, gts


def train_valid_split(img_paths, gts, valid_pct=VALID_PCT, k=None):
    num_imgs = len(img_paths)
    if k is None:
        num_valid = int(np.rint(valid_pct * num_imgs))
    else:
        # TODO:
        pass
    
    valid_idxs = np.random.choice(np.arange(num_imgs), size=num_valid, replace=False)
    train_idxs = np.setdiff1d(np.arange(num_imgs), valid_idxs)

    train_paths = img_paths[train_idxs]
    train_gts = gts[train_idxs]
    valid_paths = img_paths[valid_idxs]
    valid_gts = gts[valid_idxs]

    # TODO: Add asserts
    return train_paths, train_gts, valid_paths, valid_gts


# TODO:
def compose_augs():
    pass


def load_model(which_model=WHICH_MODEL):
    # TODO: Copy / Paste
    model.train()
    model.to(DEVICE)

    return model


def get_loss(which_loss=WHICH_LOSS):
    if which_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif which_loss == 'focal':
        raise NotImplementedError('Coming soon...')
    else:
        raise NotImplementedError('Please choose valid loss func')
    
    return criterion


def get_optimizer(which_optim=WHICH_OPTIM, model=model, lr=LR):
    if which_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif which_optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Please choose valid optimizer')
    
    return optimizer


def get_scheduler():
    # TODO: Copy / Paste
    pass


def validate(model, valid_dl):
    # TODO: Copy / Paste
    model.eval()
    with torch.no_grad():
        pass
    model.train()


def train(model, train_dl, valid_dl, criterion, optimizer, scheduler=None):
    pbar = tqdm(range(len(train_dl)*NUM_EPOCHS))
    epoch_num = 0
    itr = 0
    itr_losses = []
    losses = []
    train_dl_itr = iter(train_dl)
    print(f'Epoch Num: {epoch_num}')
    try:
        # TODO: Check this
        for _ in pbar:
            try:
                imgs, gts = next(train_dl_itr)
            except StopIteration:
                train_dl_itr = iter(train_dl)
                imgs, gts = next(train_dl_itr)
                epoch_num += 1
                print(f'Epoch Num: {epoch_num}')
            imgs = imgs.to(DEVICE)
            gts = gts.to(DEVICE)
            if WHICH_LOSS == 'bce':
                gts = gts.float()
            
            # Forward prop
            loss = criterion(model(imgs), gts)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validate
            if itr and (itr % NUM_ITERS_VALID) == 0:
                validate(model, valid_dl)

            if itr % NUM_ITERS_LOSS == 0:
                itr_losses.append(itr)
                losses.append(loss.detach().clone().cpu().numpy())

            # TODO: Saving checkpoints

            itr += 1
    except KeyboardInterrupt:
        optimizer.zero_grad()
        model.train()
    
    return model, itr_losses, losses


def plot_losses(itr_losses, losses):
    plt.plot(itr_losses, losses)
    plt.xlabel('itr')
    plt.ylabel('loss')
    plt.savefig('./train_losses.png')
    plt.clf()


def test(model, test_dl):
    # TODO: Copy / Paste
    model.eval()
    with torch.no_grad():
        pass
    model.train()


def main():
    img_paths, gts = get_data(ROOT)
    train_paths, train_gts, valid_paths, valid_gts = train_valid_split(img_paths, gts)
    
    train_ds = ScratchArcDataset(train_paths, train_gts)
    valid_ds = ScratchArcDataset(valid_paths, valid_gts)

    train_dl = DataLoader(train_ds, **TRAIN_DL_PARAMS)
    valid_dl = DataLoader(valid_ds, **VALID_DL_PARAMS)

    model = load_model()

    criterion = get_loss()
    optimizer = get_optimizer()
    if USE_SCHED:
        scheduler = get_scheduler()
    
    model, itr_losses, losses = train(model, train_dl, valid_dl, criterion, optimizer, scheduler)
    plot_losses(itr_losses, losses)

    test_paths, test_gts = get_data(ROOT_TEST)
    test_ds = ScratchArcDataset(test_paths, test_gts)
    test_dl = DataLoader(test_ds, **TEST_DL_PARAMS)

    test(model, test_dl)