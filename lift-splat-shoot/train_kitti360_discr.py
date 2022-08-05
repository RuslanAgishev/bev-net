#!/usr/bin/env python

import os
os.environ['TORCH_HOME'] = '/home/ruslan/.cache/torch/'
import torch
from torch.utils.data import random_split
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from time import time
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from src.models import compile_model, Discriminator
from src.tools import SimpleLoss, get_batch_iou, get_val_info
from src.tools import load_config, denormalize_img
from src.data import KITTI360_Map, KITTI360_MapOSM

from src.tools import get_nusc_maps
from src.data import SegmentationDataMap
from src.tools import GaussianSmoothing
from nuscenes.nuscenes import NuScenes


# Parameters
cfg = load_config("./configs/kitti360_config.yaml")
cfg_data = cfg['DATA']
cfg_train = cfg['TRAIN']

grid_conf = {
    'xbound': cfg_data['xbound'],
    'ybound': cfg_data['ybound'],
    'zbound': cfg_data['zbound'],
    'dbound': cfg_data['dbound'],
}

data_aug_conf = {
                'resize_lim': cfg_data['resize_lim'],
                'final_dim': cfg_data['final_dim'],
                'rot_lim': cfg_data['rot_lim'],
                'H': cfg_data['H'], 'W': cfg_data['W'],
                'rand_flip': cfg_data['rand_flip'],
                'bot_pct_lim': cfg_data['bot_pct_lim'],
                'cams': cfg_data['cams'],
                'Ncams': cfg_data['ncams'],
                }


def get_nuscenes_data(config_path, is_train):
    cfg_data = load_config(config_path)['DATA']
    data_aug_conf = {
        'resize_lim': cfg_data['resize_lim'],
        'final_dim': cfg_data['final_dim'],
        'rot_lim': cfg_data['rot_lim'],
        'H': cfg_data['H'], 'W': cfg_data['W'],
        'rand_flip': cfg_data['rand_flip'],
        'bot_pct_lim': cfg_data['bot_pct_lim'],
        'cams': cfg_data['cams'],
        'Ncams': cfg_data['ncams'],
    }
    grid_conf = {
        'xbound': cfg_data['xbound'],
        'ybound': cfg_data['ybound'],
        'zbound': cfg_data['zbound'],
        'dbound': cfg_data['dbound'],
    }
    nusc = NuScenes(version='v1.0-{}'.format(cfg_data['version']),
                    dataroot=cfg_data['dataroot'],
                    verbose=False)
    map_folder = cfg_data['dataroot']
    nusc_maps = get_nusc_maps(map_folder)
    valdata = SegmentationDataMap(nusc_maps,
                                  nusc,
                                  is_train=is_train,
                                  data_aug_conf=data_aug_conf,
                                  grid_conf=grid_conf)
    return valdata


def explore_data(data, data_aug_conf, samples=1, fname="data_sample.png"):

    for counter in np.random.choice(range(len(data)), samples):
        sample = data[counter]
        imgs = sample[0]
        local_map = sample[6]

        val = 0.01
        fH, fW = data_aug_conf['final_dim']
        fig = plt.figure(figsize=(2 * fW * val, (0.5 * fW + fH) * val))
        gs = mpl.gridspec.GridSpec(2, 2, height_ratios=(1.5 * fW, fH))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        plt.clf()
        for imgi, img in enumerate(imgs):
            ax = plt.subplot(gs[1 + imgi // 2, imgi % 2])
            showimg = denormalize_img(img)
            plt.imshow(showimg)
            plt.axis('off')

        ax = plt.subplot(gs[0, :])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.setp(ax.spines.values(), color='b', linewidth=2)

        # plot local map
        drivable_area = local_map[0, ...]
        cars = local_map[1, ...]
        local_map_vis = drivable_area.detach().clone()
        local_map_vis[cars.bool()] = 2.
        plt.imshow(local_map_vis.squeeze(0), cmap='Greys')
        plt.plot(local_map.size()[2] / 2., 5, 'ro', markersize=10)

        plt.xlim((local_map.size()[2], 0))
        plt.ylim((0, local_map.size()[1]))
        # plt.show()
        plt.savefig(fname)


data = KITTI360_Map(config=cfg, is_train=True)
traindata, valdata = random_split(data,
                                  [900, 88],
                                  generator=torch.Generator().manual_seed(42))
testdata = get_nuscenes_data(config_path="./configs/nuscenes_config.yaml",
                             is_train=False)

# creating small toy datasets
traindata = torch.utils.data.Subset(traindata, list(np.random.choice(range(len(traindata)), 1000)))
valdata = torch.utils.data.Subset(valdata, list(np.random.choice(range(len(valdata)), 100)))

print('Samples in train data:', len(traindata))
print('Samples in validation data:', len(valdata))

explore_data(valdata, data_aug_conf, 1, "valdata_sample.png")
explore_data(traindata, data_aug_conf, 1, "traindata_sample.png")


def worker_rnd_init(x):
    np.random.seed(13 + x)


trainloader = torch.utils.data.DataLoader(traindata,
                                          batch_size=cfg_train['batch_size'],
                                          shuffle=True,
                                          num_workers=cfg_train['num_workers'],
                                          drop_last=True,
                                          worker_init_fn=worker_rnd_init)
valloader = torch.utils.data.DataLoader(valdata,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=cfg_train['num_workers'])
testloader = torch.utils.data.DataLoader(testdata,
                                         batch_size=cfg_train['batch_size'],
                                         shuffle=False,
                                         num_workers=cfg_train['num_workers'])

# Load LSS model
device = torch.device('cpu') if cfg_train['gpuid'] < 0 else torch.device(f"cuda:{cfg_train['gpuid']}")
model = compile_model(grid_conf, data_aug_conf, outC=len(cfg_data['classes']))
# load pretrained model (opitional)
# modelf = cfg_data['pretrained_weights_path']
# print('Loading pretrained weights for LSS:', modelf)
# model.load_state_dict(torch.load(modelf))

model.to(device)
model.train()

model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])
loss_fn = SimpleLoss(cfg_train['pos_weight']).cuda(cfg_train['gpuid'])

# Discriminators
discriminator = Discriminator(num_input_channels=len(cfg_data['classes']))
discriminator.to(device)
discriminator.train()

optimizers_D = torch.optim.Adam(discriminator.parameters(), lr=cfg_train['lr_D'])

lambda_D = 0.01
xmin, xmax, dx = cfg_data['xbound']
ymin, ymax, dy = cfg_data['ybound']
occ_map_size = [int((xmax - xmin) / dx), int((ymax - ymin) / dy)]
patch = (1, occ_map_size[0] // 2 ** 4, occ_map_size[1] // 2 ** 4)

valid_1 = Variable(torch.Tensor( np.ones((cfg_train['batch_size'], *patch)) ),
                 requires_grad=False).float().cuda()
fake_0 = Variable(torch.Tensor( np.zeros((cfg_train['batch_size'], *patch)) ),
                 requires_grad=False).float().cuda()
criterion_d = nn.BCELoss()

# Train loop
dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p")
logdir = f"./runs/lss_kitti360_{dt}"
writer = SummaryWriter(logdir=logdir)

counter = 0
max_val_iou = 0.0
smoothing = GaussianSmoothing(len(cfg_data['classes']), 5, 1).to(device)

for epoch in tqdm(range(cfg_train['num_epochs'])):

    np.random.seed()
    for train_sample, test_sample in zip(trainloader, testloader):
        (imgs, rots, trans, intrins, post_rots, post_trans, local_map) = train_sample
        discr_input = test_sample[-1]

        t0 = time()
        model_optimizer.zero_grad()
        preds = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        local_map = local_map.to(device)
        discr_input = discr_input.to(device)

        real = discr_input  # [:, 1, :, :].unsqueeze(1)
        # add gaussian noise to real masks
        real = real + (0.1 ** 0.5) * torch.randn(*real.size(), device=device)
        # gaussian smoothing
        real = F.pad(real, (2, 2, 2, 2), mode='reflect')
        real = smoothing(real)
        real = real.tanh()

        fake = preds  # [:, 1, :, :].unsqueeze(1)
        # one hot encoding
        tmp = torch.zeros_like(fake, device=device)
        tmp[fake > 0.5*torch.mean(fake)] = 1.
        fake = tmp
        # add gaussian noise to real masks
        fake = fake + (0.1 ** 0.5) * torch.randn(*fake.size(), device=device)
        # gaussian smoothing
        fake = F.pad(fake, (2, 2, 2, 2), mode='reflect')
        fake = smoothing(fake)
        fake = fake.tanh()

        # swap fake and real ocassionaly
        if torch.rand(1) < 0.33:
            real, fake = fake, real

        # print('Real', torch.unique(real))
        # print('Fake', torch.unique(fake.detach()))

        fake_pred = discriminator(fake)
        real_pred = discriminator(real)

        loss_GAN = criterion_d(fake_pred, valid_1)

        loss_D = criterion_d(fake_pred, fake_0) + criterion_d(real_pred, valid_1)
        loss = loss_fn(preds, local_map)
        loss_G = lambda_D * loss_GAN + loss

        if epoch >= cfg_train['discr_start_train_epoch']:
            # training discriminator and the model
            loss_G.backward(retain_graph=True)
            optimizers_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['max_grad_norm'])
            model_optimizer.step()
            optimizers_D.step()
        else:
            # training only the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['max_grad_norm'])
            model_optimizer.step()

        counter += 1
        t1 = time()

        # cv2.imshow('Real', real[0][0].cpu().numpy())
        # cv2.imshow('Fake', fake.detach()[0][0].cpu().numpy())
        # cv2.waitKey(3)

        if counter % 10 == 0:
            print(counter, loss.item())
            writer.add_scalar('train/loss', loss, counter)
            writer.add_scalar('train/discriminator_loss', loss_D, counter)
            writer.add_scalar('train/generator_loss', loss_GAN, counter)

            # Discriminator accuracy
            fake_accuracy = ((fake_pred > 0.5) == 0).float().mean()
            real_accuracy = ((real_pred > 0.5) == 1).float().mean()
            writer.add_scalar('train/fake_accuracy', fake_accuracy, counter)
            writer.add_scalar('train/real_accuracy', real_accuracy, counter)

        if counter % 50 == 0:
            _, _, iou = get_batch_iou(preds, local_map)
            _, _, iou_static = get_batch_iou(preds[:, 0, :, :], local_map[:, 0, :, :])
            _, _, iou_dynamic = get_batch_iou(preds[:, 1, :, :], local_map[:, 1, :, :])

            # writer.add_scalar('train/iou', iou, counter)
            writer.add_scalar('train/iou_static', iou_static, counter)
            writer.add_scalar('train/iou_dynamic', iou_dynamic, counter)
            writer.add_scalar('train/epoch', epoch, counter)
            # writer.add_scalar('train/step_time', t1 - t0, counter)

        if counter % 50 == 0:
            val_info = get_val_info(model, valloader, loss_fn, device)
            print('VAL', val_info)
            writer.add_scalar('val/loss', val_info['loss'], counter)
            writer.add_scalar('val/iou', val_info['iou'], counter)
            writer.add_scalar('val/iou_static', val_info['iou_static'], counter)
            writer.add_scalar('val/iou_dynamic', val_info['iou_dynamic'], counter)

            if val_info['iou'] > max_val_iou:
                max_val_iou = val_info['iou']
                model.eval()
                mname = os.path.join(logdir, "model_iou_{:.2f}.pt".format(val_info['iou']))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
