import pytorch_lightning as pl
import torch
from dataloader import *
from dataloader_comma import *
from dataloader_nuscenes import * 
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from utils import pad_collate
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LaneModule(pl.LightningModule):
    '''Pytorch lightning module to train angle, distance or multitask procedures'''
    def __init__(self, model, bs, multitask="multitask", dataset="comma", time_horizon=3, ground_truth="normal", intervention=False, dataset_path=None, dataset_fraction=1.0):
        super(LaneModule, self).__init__()
        self.dataset_fraction = dataset_fraction
        self.model = model
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.intervention = intervention
        self.dataset_path = dataset_path
        self.num_workers = 4
        self.multitask = multitask
        self.bs = bs
        self.time_horizon = time_horizon
        self.loss = self.mse_loss
        #self.save_hyperparameters(ignore=['model'])
        self.bce_loss = nn.BCELoss()

    def forward(self, x, angle, distance, vego):
        return self.model(x, angle, distance, vego)

    def mse_loss(self, input, target, mask, reduction="mean"):
        input = input.float()
        target = target.float()

        out = (input[~mask]-target[~mask])**2
        return out.mean() if reduction == "mean" else out 

    def calculate_loss(self, logits, angle, distance):
        sm = nn.Softmax(dim=1)
        if self.multitask == "multitask":
            #ho messo logits[0] perchè il forward restituisce una tupla, prima era logits
            # (x[:,1:F+1,:], x2[:,1:F+1,:],self.multitask_param_angle, self.multitask_param_dist), attentions -G.R.
            logits_angle, logits_dist, param_angle, param_dist = logits[0]
            mask = distance.squeeze() == 0.0
            #TODO: The intervention Boolean flag indicates whether 
            # a special type of supervision is being used in which:
            # • Instead of predicting the actual steering angle,
            # • The model must predict where the driver should have intervened (intervention label), as if it were an "alert system."
            if not self.intervention:
                loss_angle = torch.sqrt(self.loss(logits_angle.squeeze(), angle.squeeze(), mask))
            else: 
                angle, distance = distance, angle
                mask = distance.squeeze() == 0.0
                loss_angle = self.bce_loss(sm(logits_angle.float()).squeeze()[~mask], angle.float().squeeze()[~mask])
            loss_distance = torch.sqrt(self.loss(logits_dist.squeeze(), distance.squeeze(), mask))
            if loss_angle.isnan() or loss_distance.isnan():
                print("ERROR")
            loss = loss_angle, loss_distance
            self.log_dict({"train_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs,sync_dist=True)
            self.log_dict({"train_loss_distance": loss_distance}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            return loss_angle, loss_distance, param_angle, param_dist
        else:
            target = angle if self.multitask == "angle" else distance
            mask = distance.squeeze() == 0.0
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss = torch.sqrt(self.loss(logits.squeeze(), target.squeeze(), mask))
            #e lui ha insierito questo:
            #----------------------------------------------
            #logits_tensor = logits[0] if isinstance(logits, tuple) else logits
            #loss = torch.sqrt(self.loss(logits_tensor.squeeze(), angle.squeeze(), mask))
            #----------------------------------------------
            return loss

    def training_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits, attns = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            #0.3 and 0.7 hyperparameters used to give more importance to distance prediction than angle prediction -G.R.
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        self.log_dict({"train_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
       #TODO: time_horizon is an integer parameter that indicates how often the model 
       # should make a prediction during inference or testing. It is used to simulate 
       # a sequential prediction over time, as if the vehicle were proceeding into the 
       # future, prediction after prediction, using previous outputs as inputs.
        if self.time_horizon > 1:

            logits_all = []
            for i in range(self.time_horizon, vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon):
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle[:,0:i+j], distance[:,0:i+j]
                    if self.multitask == "angle" and len(logits_all) > 0:
                        angle[:,i+j] = torch.tensor(logits_all)[-1]
                    if self.multitask == "distance" and len(logits_all) > 0:
                        distance[:,i+j] = torch.tensor(logits_all)[-1]
                    if self.multitask == "multitask":
                        logits, attns = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)
                        logits = logits[0][:, -1], logits[1][:, -1]
                    else:
                        logits, attns = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)
                        logits = logits[:, -1]
                    logits_all.append(logits)
            return torch.tensor(logits_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:]

        
        res, probs = self(image_array, angle, distance, vego)
        return res, angle, distance

    def validation_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits, probs = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        self.log_dict({"val_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        if self.time_horizon > 1:
            logits_all = []
            for i in range(self.time_horizon, vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon):
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle[:,0:i+j], distance[:,0:i+j]
                    if self.multitask == "angle" and len(logits_all) > 0:
                        angle[:, i+j] = logits_all[-1].squeeze()
                    if self.multitask == "distance" and len(logits_all) > 0:
                        distance[:, i+j] = logits_all[-1].squeeze()
                    if self.multitask == "multitask":
                        logits, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)
                        logits = logits[0][:, -1], logits[1][:, -1]
                    else:
                        res, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego)
                        logits, attns = res
                        logits = logits[:, -1]
                    logits_all.append(logits)
            loss = self.calculate_loss(torch.tensor(logits_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:])
            self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            return loss
    
        _, image_array, vego, angle, distance, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        logits, attns = self(image_array, angle, distance, vego)
        loss = self.calculate_loss(logits, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"test_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"test_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        return loss
    #------------------------------------------------------------

    def train_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log_dict({"train_loss_accumulated": losses }, batch_size=self.bs, sync_dist=True)

    def validation_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"val_loss_accumulated": losses }, batch_size=self.bs, sync_dist=True)

    def test_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"test_loss_accumulated": losses }, batch_size=self.bs, sync_dist=True)

    #------------------------------------------------------------
    def train_dataloader(self):
        return self.get_dataloader(dataset_type="train")

    def val_dataloader(self):
        return self.get_dataloader(dataset_type="val")

    def test_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def predict_dataloader(self):
        return self.get_dataloader(dataset_type="test")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-5)
        return g_opt

    def get_dataloader(self, dataset_type):
        if self.dataset == "once":
            ds = ONCEDataset(dataset_type=dataset_type, multitask=self.multitask) 
        elif self.dataset == "comma":
            ds = CommaDataset(dataset_type=dataset_type, multitask=self.multitask if not self.intervention else "intervention", ground_truth=self.ground_truth, dataset_path=self.dataset_path, dataset_fraction=self.dataset_fraction)
        elif self.dataset == 'nuscenes':
            ds = NUScenesDataset(dataset_type=dataset_type, multitask=self.multitask if not self.intervention else "intervention", ground_truth=self.ground_truth, max_len=20, dataset_path=self.dataset_path, dataset_fraction=self.dataset_fraction)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)