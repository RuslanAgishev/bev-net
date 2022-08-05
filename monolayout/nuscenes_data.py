from nuscenes_src.data import SegmentationDataMap
from nuscenes.nuscenes import NuScenes
from nuscenes_src.tools import get_nusc_maps
from utils import load_config
import torch
from monolayout.datasets import process_discr
import PIL.Image as pil
import os
import numpy as np


cfg_data = load_config("nuscenes_config.yaml")['DATA']
cfg_train = load_config("nuscenes_config.yaml")['TRAIN']

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

class SegmentationDataMapOSM(SegmentationDataMap):
    def __init__(self, model_type, osm_path, *args, **kwargs):
        super(SegmentationDataMapOSM, self).__init__(*args, **kwargs)
        self.occ_map_size = int((grid_conf['xbound'][1]-grid_conf['xbound'][0])/grid_conf['xbound'][2])
        self.model_type = model_type
        self.data_aug_conf = data_aug_conf
        self.osm_path = osm_path

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        dynamic_binimg = self.get_binimg(rec)
        static_binimg = self.get_local_map_torch(rec)
        map_binimg = torch.cat([static_binimg, dynamic_binimg], dim=0)

        discr = {}
        if self.is_train:
            if self.model_type == 'dynamic':
                dynamic_binimg_pil = pil.fromarray(dynamic_binimg.cpu().numpy().squeeze())
                dynamic_discr = process_discr(dynamic_binimg_pil, self.occ_map_size)
                dynamic_discr = torch.transpose(torch.Tensor(dynamic_discr), 2, 0)
                discr['dynamic'] = dynamic_discr
            elif self.model_type == 'static':
                static_discr = process_discr(self.get_osm(self.osm_path), self.occ_map_size)
                static_discr = torch.transpose(torch.Tensor(static_discr), 2, 0)
                discr['static'] = static_discr
            elif self.model_type == 'both':
                # input for static (road) objects discriminator
                static_discr = process_discr(self.get_osm(self.osm_path), self.occ_map_size)
                static_discr = torch.transpose(torch.Tensor(static_discr), 2, 0)
                discr['static'] = static_discr
                # input for dynamic (cars) objects discriminator
                dynamic_binimg_pil = pil.fromarray(dynamic_binimg.cpu().numpy().squeeze())
                dynamic_discr = process_discr(dynamic_binimg_pil, self.occ_map_size)
                dynamic_discr = torch.transpose(torch.Tensor(dynamic_discr), 2, 0)
                discr['dynamic'] = dynamic_discr

        return imgs, rots, trans, intrins, post_rots, post_trans, map_binimg, discr

    @staticmethod
    def get_osm(root_dir):
        # get_osm_path
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)
        with open(osm_path, 'rb') as f:
            with pil.open(f) as img:
                return img.convert('RGB')



def get_nuscenes_datasets():
    nusc = NuScenes(version='v1.0-{}'.format(cfg_data['version']),
                    dataroot=cfg_data['dataroot'],
                    verbose=False)
    map_folder = cfg_data['dataroot']
    nusc_maps = get_nusc_maps(map_folder)
    model_type = cfg_train['type']
    osm_path = cfg_train['osm_path']

    valdata = SegmentationDataMapOSM(model_type=model_type,
                                     osm_path=osm_path,
                                     nusc_maps=nusc_maps,
                                     nusc=nusc,
                                     is_train=False,
                                     data_aug_conf=data_aug_conf,
                                     grid_conf=grid_conf)
    traindata = SegmentationDataMapOSM(model_type=model_type,
                                       osm_path=osm_path,
                                       nusc_maps=nusc_maps,
                                       nusc=nusc,
                                       is_train=True,
                                       data_aug_conf=data_aug_conf,
                                       grid_conf=grid_conf)
    return traindata, valdata
