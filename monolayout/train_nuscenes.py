import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_HOME'] = '/home/ruslan/.cache/torch/'
import monolayout
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import tqdm
from utils import mean_IU, mean_precision, load_config
from nuscenes_data import get_nuscenes_datasets
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime


def save_img_np(img_np, name_dest_im):
    img_np = 255 * (img_np / np.max(img_np))
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, img_np)
    # print("Saved img to {}".format(name_dest_im))


class Trainer:
    def __init__(self, config_fname='./nuscenes_config.yaml'):
        self.cfg = load_config(config_fname)
        self.models = {}
        self.weight = {}
        self.weight["static"] = self.cfg['TRAIN']['static_weight']
        self.weight["dynamic"] = self.cfg['TRAIN']['dynamic_weight']
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.occ_map_size = int(
            (self.cfg['DATA']['xbound'][1] - self.cfg['DATA']['xbound'][0]) / self.cfg['DATA']['xbound'][2])

        dt = datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p")
        self.save_path = f"./runs/{self.cfg['TRAIN']['model_name']}_{self.cfg['TRAIN']['type']}_{dt}"
        self.writer = SummaryWriter(logdir=self.save_path)
        self.counter = 0

        # Initializing models
        self.models["encoder"] = monolayout.Encoder(
            18, self.cfg['DATA']['final_dim'][0], self.cfg['DATA']['final_dim'][1], True)
        if self.cfg['TRAIN']['type'] == "both":
            self.models["static_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["dynamic_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["static_discr"] = monolayout.Discriminator()
        else:
            self.models["decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["discriminator"] = monolayout.Discriminator()

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(
                    self.models[key].parameters())
            else:
                self.parameters_to_train += list(self.models[key].parameters())

        # Optimization
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.cfg['TRAIN']['lr'])
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.cfg['TRAIN']['scheduler_step_size'], 0.1)

        self.model_optimizer_D = optim.Adam(
            self.parameters_to_train_D, self.cfg['TRAIN']['lr'])
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.model_optimizer_D, self.cfg['TRAIN']['scheduler_step_size'], 0.1)

        self.patch = (1, self.occ_map_size // 2 ** 4, self.occ_map_size // 2 ** 4)

        self.valid = Variable(
                        torch.Tensor( np.ones((self.cfg['TRAIN']['batch_size'], *self.patch)) ),
                        requires_grad=False).float().cuda()
        self.fake = Variable(
                        torch.Tensor( np.zeros((self.cfg['TRAIN']['batch_size'], *self.patch)) ),
                        requires_grad=False).float().cuda()

        # Data Loaders
        train_dataset, val_dataset = get_nuscenes_datasets()

        self.train_loader = DataLoader(
            train_dataset,
            self.cfg['TRAIN']['batch_size'],
            True,
            num_workers=self.cfg['TRAIN']['num_workers'],
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.cfg['TRAIN']['num_workers'],
            pin_memory=True,
            drop_last=True)

        if self.cfg['TRAIN']['load_weights_folder'] != "":
            self.load_model()

        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):

        for self.epoch in range(self.cfg['TRAIN']['num_epochs']):
            loss = self.run_epoch()
            print("Epoch: %d | Loss: %.4f |" %
                  (self.epoch, loss["loss"]))

            if self.epoch % self.cfg['TRAIN']['log_frequency'] == 0:
                self.validation()
                self.save_model()

    def process_batch(self, data, validation=False):
        outputs = {}
        inputs = {}
        inputs['color'] = data[0].squeeze(1).to(self.device)
        inputs['static'] = data[6][:, 0, :, :].to(self.device)
        inputs['dynamic'] = data[6][:, 1, :, :].to(self.device)

        features = self.models["encoder"](inputs["color"])

        if self.cfg['TRAIN']['type'] == "both":
            outputs["dynamic"] = self.models["dynamic_decoder"](features)
            outputs["static"] = self.models["static_decoder"](features)
        else:
            outputs["topview"] = self.models["decoder"](features)
        if validation:
            return outputs
        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0.0, 0.0
        for batch_idx, data in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(data)
            self.model_optimizer.zero_grad()

            fake_pred = self.models["discriminator"](outputs["topview"])
            discr_input = data[7][self.cfg['TRAIN']['type']].to(self.device)
            real_pred = self.models["discriminator"](discr_input.float())
            loss_GAN = self.criterion_d(fake_pred, self.valid)
            loss_D = self.criterion_d(
                fake_pred, self.fake) + self.criterion_d(real_pred, self.valid)
            loss_G = self.cfg['TRAIN']['lambda_D'] * loss_GAN + losses["loss"]

            # Train Discriminator
            if self.epoch > self.cfg['TRAIN']['discr_train_epoch']:
                loss_G.backward(retain_graph=True)
                self.model_optimizer_D.zero_grad()
                loss_D.backward()
                self.model_optimizer.step()
                self.model_optimizer_D.step()

                # tensorboard logging
                if self.counter % 10 == 0:
                    # print(f'Discr loss: {loss_D.item()}')
                    self.writer.add_scalar('discrimimator/loss', loss_D.item(), self.counter)
            else:
                losses["loss"].backward()
                self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += loss_D.item()

        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)

        return loss

    def validation(self, img_save_interval=10):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
        iou_static, mAP_static = np.array([0., 0.]), np.array([0., 0.])
        iou_dynamic, mAP_dynamic = np.array([0., 0.]), np.array([0., 0.])
        for batch_idx, data in tqdm.tqdm(enumerate(self.val_loader)):
            inputs = {}
            inputs['color'] = data[0].squeeze(1)
            inputs['static'] = data[6][:, 0, :, :]
            inputs['dynamic'] = data[6][:, 1, :, :]
            with torch.no_grad():
                outputs = self.process_batch(data, True)

                if self.cfg['TRAIN']['type'] == "both":
                    pred_dynamic = np.squeeze(torch.argmax(outputs["dynamic"].detach(), 1).cpu().numpy())
                    true_dynamic = np.squeeze(inputs["dynamic"].detach().cpu().numpy())
                    iou_dynamic_batch = mean_IU(pred_dynamic, true_dynamic)
                    mAP_dynamic_batch = mean_precision(pred_dynamic, true_dynamic)

                    pred_static = np.squeeze(torch.argmax(outputs["static"].detach(), 1).cpu().numpy())
                    true_static = np.squeeze(inputs["static"].detach().cpu().numpy())
                    iou_static_batch = mean_IU(pred_static, true_static)
                    mAP_static_batch = mean_precision(pred_static, true_static)

                    # save input image and static objects predictions
                    if batch_idx % img_save_interval == 0:
                        self.writer.add_image(f'val/color_input/{batch_idx}',
                                              inputs['color'].squeeze(),
                                              self.counter)
                        self.writer.add_image(f'val/pred_dynamic/{batch_idx}',
                                                  outputs["dynamic"].squeeze(),
                                                  self.counter)
                        self.writer.add_image(f'val/pred_static/{batch_idx}',
                                                  outputs["static"].squeeze(),
                                                  self.counter)
                    iou_static += iou_static_batch
                    mAP_static += mAP_static_batch

                    iou_dynamic += iou_dynamic_batch
                    mAP_dynamic += mAP_dynamic_batch

                    iou += (np.array(iou_static_batch) + np.array(iou_dynamic_batch)) / 2.
                    mAP += (np.array(mAP_static_batch) + np.array(mAP_dynamic)) / 2.

                else:
                    pred = np.squeeze( torch.argmax(outputs["topview"].detach(), 1 ).cpu().numpy())
                    true = np.squeeze( inputs[self.cfg['TRAIN']['type']].detach().cpu().numpy() )

                    iou += mean_IU(pred, true)
                    mAP += mean_precision(pred, true)

        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)

        # logging
        if self.cfg['TRAIN']['type'] == "both":
            iou_static /= len(self.val_loader)
            mAP_static /= len(self.val_loader)
            iou_dynamic /= len(self.val_loader)
            mAP_dynamic /= len(self.val_loader)
            self.writer.add_scalar('val/iou_static', iou_static[1], self.counter)
            self.writer.add_scalar('val/mAP_static', mAP_static[1], self.counter)

            self.writer.add_scalar('val/iou_dynamic', iou_dynamic[1], self.counter)
            self.writer.add_scalar('val/mAP_dynamic', mAP_dynamic[1], self.counter)

        self.writer.add_scalar('val/iou', iou[1], self.counter)
        self.writer.add_scalar('val/mAP', mAP[1], self.counter)
        print("Epoch: %d | Validation: mIOU: %.4f mAP: %.4f" %
              (self.epoch, iou[1], mAP[1]))

    def compute_losses(self, inputs, outputs, logger_mode='train'):
        losses = {}
        if self.cfg['TRAIN']['type'] == "both":
            losses["static_loss"] = self.compute_topview_loss(
                                            outputs["static"],
                                            inputs["static"],
                                            self.weight['static'])
            losses["dynamic_loss"] = self.compute_topview_loss(
                                            outputs["dynamic"],
                                            inputs["dynamic"],
                                            self.weight['dynamic'])

            losses["loss"] = (losses["static_loss"] + losses["dynamic_loss"]) / 2.

            # tensorboard logging
            self.counter += 1
            if self.counter % 10 == 0:
                print( f"Step: %d | {logger_mode} loss: %.4f" %
                       (self.counter, losses["loss"].item()) )
                self.writer.add_scalar(f'{logger_mode}/static_loss', losses["static_loss"], self.counter)
                self.writer.add_scalar(f'{logger_mode}/dynamic_loss', losses["dynamic_loss"], self.counter)
                self.writer.add_scalar(f'{logger_mode}/loss', losses["loss"], self.counter)
        else:
            losses["loss"] = self.compute_topview_loss(
                                            outputs["topview"],
                                            inputs[self.cfg['TRAIN']['type']],
                                            self.weight[self.cfg['TRAIN']['type']])
            # tensorboard logging
            self.counter += 1
            if self.counter % 10 == 0:
                print(self.counter, losses["loss"].item())
                self.writer.add_scalar(f'{logger_mode}/loss', losses["loss"], self.counter)

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):

        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(
            self.save_path,
            f"nuscenes_{self.cfg['TRAIN']['type']}",
            f"weights_{self.epoch}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.cfg['DATA']['final_dim'][0]
                state_dict["width"] = self.cfg['DATA']['final_dim'][1]

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.cfg['TRAIN']['load_weights_folder'] = os.path.expanduser(
            self.cfg['TRAIN']['load_weights_folder'])

        assert os.path.isdir(self.cfg['TRAIN']['load_weights_folder']), \
            "Cannot find folder {}".format(self.cfg['TRAIN']['load_weights_folder'])
        print(
            "loading model from folder {}".format(
                self.cfg['TRAIN']['load_weights_folder']))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.cfg['TRAIN']['load_weights_folder'],
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.cfg['TRAIN']['load_weights_folder'], "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

