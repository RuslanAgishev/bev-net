#!/usr/bin/env python
import os

from src.data import SegmentationDataMapOSM
from nuscenes.nuscenes import NuScenes

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.models import compile_model
from src.models import Discriminator
from src.tools import SimpleLoss
from src.tools import get_batch_iou
from src.tools import get_val_info
from src.tools import load_config
from src.tools import get_nusc_maps


# Parameters
cfg_data = load_config("./nuscenes_config.yaml")['DATA']
cfg_train = load_config("./nuscenes_config.yaml")['TRAIN']

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


# Load data
nusc = NuScenes(version='v1.0-{}'.format(cfg_data['version']),
                dataroot=cfg_data['dataroot'],
                verbose=False)

map_folder = cfg_data['dataroot']
nusc_maps = get_nusc_maps(map_folder)
osm_path = cfg_data['osm_path']

valdata = SegmentationDataMapOSM(osm_path,
                                 nusc_maps,
                                 nusc,
                                 is_train=False,
                                 data_aug_conf=data_aug_conf,
                                 grid_conf=grid_conf)
traindata = SegmentationDataMapOSM(osm_path,
                                   nusc_maps,
                                   nusc,
                                   is_train=True,
                                   data_aug_conf=data_aug_conf,
                                   grid_conf=grid_conf)

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


# Define LSS model
device = torch.device('cpu') if cfg_train["gpuid"] < 0 else torch.device(f'cuda:{cfg_train["gpuid"]}')

model = compile_model(grid_conf, data_aug_conf, outC=2)
model.to(device)
model_optimizer = torch.optim.Adam(model.parameters(),
                                   lr=cfg_train['lr'],
                                   weight_decay=cfg_train['weight_decay'])
loss_fn = SimpleLoss(cfg_train['pos_weight']).cuda(cfg_train["gpuid"])
model.train()


# Discriminator
from torch.autograd import Variable
import torch.nn as nn

discriminators = {}
discriminators['static'] = Discriminator()
discriminators['dynamic'] = Discriminator()

for key in discriminators.keys():
    discriminators[key].to(device)
    discriminators[key].train()

optimizers_D = {}
optimizers_D['static'] = torch.optim.Adam(discriminators['static'].parameters(), lr=cfg_train['lr_D'])
optimizers_D['dynamic'] = torch.optim.Adam(discriminators['dynamic'].parameters(), lr=cfg_train['lr_D'])

weights_D = {}
weights_D['static'] = cfg_train['static_weight_D']
weights_D['dynamic'] = cfg_train['dynamic_weight_D']

lambda_D = 0.01
occ_map_size = int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / grid_conf['xbound'][2])
patch = (1, occ_map_size // 2 ** 4, occ_map_size // 2 ** 4)

valid = Variable(torch.Tensor( np.ones((cfg_train['batch_size'], *patch)) ),
                 requires_grad=False).float().cuda()
fake = Variable(torch.Tensor( np.zeros((cfg_train['batch_size'], *patch)) ),
                requires_grad=False).float().cuda()
criterion_d = nn.BCEWithLogitsLoss()


# Training loop
dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p")
logdir = f"./runs/{cfg_train['model_name']}_{dt}"
writer = SummaryWriter(logdir=logdir)

counter = 0
max_val_iou = 0.0

for epoch in tqdm(range(cfg_train['num_epochs'])):
    np.random.seed()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, discr_input) in enumerate(trainloader):
        t0 = time()
        model_optimizer.zero_grad()
        preds = model(imgs.to(device),
                      rots.to(device),
                      trans.to(device),
                      intrins.to(device),
                      post_rots.to(device),
                      post_trans.to(device),
                      )
        binimgs = binimgs.to(device)
        static_discr_input = discr_input[:, 0, :, :].unsqueeze(1).to(device)
        dynamic_discr_input = discr_input[:, 1, :, :].unsqueeze(1).to(device)

        loss = loss_fn(preds, binimgs)

        fake_pred_static = discriminators['static'](preds[:, 0, :, :].unsqueeze(1))
        real_pred_static = discriminators['static'](static_discr_input)

        fake_pred_dynamic = discriminators['dynamic'](preds[:, 1, :, :].unsqueeze(1))
        real_pred_dynamic = discriminators['dynamic'](dynamic_discr_input)

        loss_GAN = weights_D['static'] * criterion_d(fake_pred_static, valid) + \
                   weights_D['dynamic'] * criterion_d(fake_pred_dynamic, valid)

        loss_D_static = weights_D['static'] * (criterion_d(fake_pred_static, fake) + \
                                               criterion_d(real_pred_static, valid))
        loss_D_dynamic = weights_D['dynamic'] * (criterion_d(fake_pred_dynamic, fake) + \
                                                 criterion_d(real_pred_dynamic, valid))

        loss_D = loss_D_static + loss_D_dynamic
        loss_G = lambda_D * loss_GAN + loss

        if epoch > cfg_train['discr_start_train_epoch']:
            loss_G.backward(retain_graph=True)
            optimizers_D['static'].zero_grad()
            optimizers_D['dynamic'].zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['max_grad_norm'])
            model_optimizer.step()
            optimizers_D['static'].step()
            optimizers_D['dynamic'].step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['max_grad_norm'])
            model_optimizer.step()

        counter += 1
        t1 = time()

        if counter % 10 == 0:
            writer.add_scalar('train/loss', loss, counter)
            writer.add_scalar('discriminator/loss', loss_D, counter)
            writer.add_scalar('discriminator/loss_static', loss_D_static, counter)
            writer.add_scalar('discriminator/loss_dynamic', loss_D_dynamic, counter)

        if counter % 50 == 0:
            _, _, iou = get_batch_iou(preds, binimgs)
            writer.add_scalar('train/iou', iou, counter)
            writer.add_scalar('train/epoch', epoch, counter)
            writer.add_scalar('train/step_time', t1 - t0, counter)

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
                mname = os.path.join(logdir, "model_iou_{}.pt".format(val_info['iou']))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
