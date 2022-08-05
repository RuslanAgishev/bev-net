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
from torch.autograd import Variable
import torch.nn as nn

from src.models import compile_model
from src.models import Discriminator
from src.tools import SimpleLoss
from src.tools import get_batch_iou
from src.tools import get_val_info
from src.tools import load_config
from src.tools import get_nusc_maps
from src.data import compile_data


# Parameters
cfg_data = load_config("./configs/nuscenes_config.yaml")['DATA']
cfg_train = load_config("./configs/nuscenes_config.yaml")['TRAIN']

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
trainloader, valloader = compile_data(cfg_data['version'],
                                      cfg_data['dataroot'],
                                      data_aug_conf=data_aug_conf,
                                      grid_conf=grid_conf,
                                      bsz=cfg_train['batch_size'],
                                      nworkers=cfg_train['num_workers'],
                                      parser_name='segmentationdatamap')

# Define LSS model
device = torch.device('cpu') if cfg_train["gpuid"] < 0 else torch.device(f'cuda:{cfg_train["gpuid"]}')
model = compile_model(grid_conf, data_aug_conf, outC=2)
model.to(device)
model_optimizer = torch.optim.Adam(model.parameters(),
                                   lr=cfg_train['lr'],
                                   weight_decay=cfg_train['weight_decay'])
loss_fn = SimpleLoss(cfg_train['pos_weight']).cuda(cfg_train["gpuid"])
model.train()


# Training loop
dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p")
logdir = f"./runs/{cfg_train['model_name']}_{dt}"
writer = SummaryWriter(logdir=logdir)

counter = 0
max_val_iou = 0.0

for epoch in tqdm(range(cfg_train['num_epochs'])):
    np.random.seed()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
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

        loss = loss_fn(preds, binimgs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train['max_grad_norm'])
        model_optimizer.step()

        counter += 1
        t1 = time()
        
        if counter % 10 == 0:
            print(counter, loss.item())
            writer.add_scalar('train/loss', loss, counter)

        if counter % 50 == 0:
            _, _, iou_static = get_batch_iou(preds[:, 0, :, :], binimgs[:, 0, :, :])
            _, _, iou_dynamic = get_batch_iou(preds[:, 1, :, :], binimgs[:, 1, :, :])
            _, _, iou = get_batch_iou(preds, binimgs)
            writer.add_scalar('train/iou_static', iou_static, counter)
            writer.add_scalar('train/iou_dynamic', iou_dynamic, counter)
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
