import pytorch_lightning as pl
import torch
from dataloader_comma import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn 
from utils import pad_collate
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class LaneModule(pl.LightningModule):
    '''Pytorch lightning module to train angle, distance or multitask procedures'''
    def __init__(self, model, bs, multitask="multitask", dataset="comma", time_horizon=1, ground_truth="normal", intervention=False, dataset_path=None, dataset_fraction=1.0):
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

    def forward(self, x, angle, distance, vego, seq_key):
        return self.model(x, angle, distance, vego, seq_key)
    
    def mae_loss(self, input, target, mask):
        input = input.float()
        target = target.float()
        out = torch.abs(input[~mask] - target[~mask])
        return out.mean()

    def mse_loss(self, input, target, mask, reduction="mean"):
        input = input.float()
        target = target.float()
        out = (input[~mask]-target[~mask])**2
        return out.mean() if reduction == "mean" else out 
    
    def calculate_mae_loss(self, res, angle, distance):
        
        if self.multitask == "multitask":
            
            # (x[:,1:F+1,:], x2[:,1:F+1,:],self.multitask_param_angle, self.multitask_param_dist), attentions -G.R.
            logits_angle, logits_dist, param_angle, param_dist = res[0]

            mask_angle = angle.squeeze() == 0.0
            mask_distance = distance.squeeze() == 0.0

            #The intervention Boolean flag indicates whether 
            # a special type of supervision is being used in which:
            # • Instead of predicting the actual steering angle,
            # • The model must predict where the driver should have intervened (intervention label), as if it were an "alert system."
            if not self.intervention:
                loss_angle = self.mae_loss(logits_angle.squeeze(), angle.squeeze(), mask_angle)
            else: 
                sm = nn.Softmax(dim=1)
                angle, distance = distance, angle
                loss_angle = self.bce_loss(sm(logits_angle.float()).squeeze()[~mask_angle], angle.float().squeeze()[~mask_angle])
            #RMSE (root-mean-square error) -G.R.
            loss_distance = self.mae_loss(logits_dist.squeeze(), distance.squeeze(), mask_distance)

            if loss_angle.isnan() or loss_distance.isnan():
                print("ERROR")
                
            loss = loss_angle, loss_distance
            return loss_angle, loss_distance, param_angle, param_dist
        else:
            target = angle if self.multitask == "angle" else distance
            mask = target.squeeze() == 0.0

            logits = res[0] if isinstance(res, tuple) else res
            #RMSE (root-mean-square error) -G.R.
            loss = self.mae_loss(logits.squeeze(), target.squeeze(), mask)
            return loss

    def calculate_loss(self, res, angle, distance):
        
        if self.multitask == "multitask":
            
            # (x[:,1:F+1,:], x2[:,1:F+1,:],self.multitask_param_angle, self.multitask_param_dist), attentions -G.R.
            logits_angle, logits_dist, param_angle, param_dist = res[0]
            mask_angle = angle.squeeze() == 0.0
            mask_distance = distance.squeeze() == 0.0
            if not self.intervention:
                loss_angle = torch.sqrt(self.loss(logits_angle.squeeze(), angle.squeeze(), mask_angle))
            else: 
                sm = nn.Softmax(dim=1)
                angle, distance = distance, angle
                loss_angle = self.bce_loss(sm(logits_angle.float()).squeeze()[~mask_angle], angle.float().squeeze()[~mask_angle])
            #RMSE (root-mean-square error) -G.R.
            loss_distance = torch.sqrt(self.loss(logits_dist.squeeze(), distance.squeeze(), mask_distance))

            if loss_angle.isnan() or loss_distance.isnan():
                print("ERROR")
                
            loss = loss_angle, loss_distance
            return loss_angle, loss_distance, param_angle, param_dist
        else:
            target = angle if self.multitask == "angle" else distance
            mask = target.squeeze() == 0.0
            logits = res[0] if isinstance(res, tuple) else res
            #RMSE (root-mean-square error) -G.R.
            loss = torch.sqrt(self.loss(logits.squeeze(), target.squeeze(), mask))
            if loss.isnan():
                print("ERROR")
            return loss

    def training_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, seq_key, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        res, probs = self(image_array, angle, distance, vego, seq_key)
    
        loss = self.calculate_loss(res, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            #0.3 and 0.7 hyperparameters used to give more importance to distance prediction than angle prediction -G.R.
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist) 
            
            self.log_dict({"train_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"train_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)

        self.log_dict({"train_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, seq_key, m_lens, i_lens, s_lens, a_lens, d_lens = batch
       #Time horizon is an integer parameter that indicates how often the model 
       # should make a prediction during inference or testing. It is used to simulate 
       # a sequential prediction over time, as if the vehicle were proceeding into the 
       # future, prediction after prediction, using previous outputs as inputs.
        if self.time_horizon > 1:
            logits_all = []
            logits_angle_all = []
            logits_distance_all = []
            attns_all = []
            angle_autoreg = angle.clone()
            distance_autoreg = distance.clone()
            for i in range(self.time_horizon, vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon):
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle_autoreg[:,0:i+j], distance_autoreg[:,0:i+j]
                    if self.multitask == "angle" and len(logits_all) > 0:
                        angle_autoreg[:, i+j] = logits_all[-1].squeeze(-1)
                    if self.multitask == "distance" and len(logits_all) > 0:
                        distance_autoreg[:, i+j] = logits_all[-1].squeeze(-1)
                    if self.multitask == "multitask":
                        res, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego, seq_key)
                        logits, attns = res
                        param_angle, param_dist= logits[2], logits[3]
                        logits_angle, logits_distance = logits[0][:, -1], logits[1][:, -1]
                        angle_autoreg[:, i+j] = logits_angle.squeeze(-1)
                        distance_autoreg[:, i+j] = logits_distance.squeeze(-1)
                        logits_angle_all.append(logits_angle)
                        logits_distance_all.append(logits_distance)
                    else:
                        res, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego, seq_key)
                        logits, attns = res
                        logits = logits[:, -1]
                        logits_all.append(logits)
                    attns_all.append(attns if attns is not None else torch.zeros_like(attns))

            if self.multitask == "multitask":
                logits_angle_all = torch.stack(logits_angle_all, dim=1)
                logits_distance_all = torch.stack(logits_distance_all, dim=1)
                res = ((logits_angle_all, logits_distance_all, param_angle, param_dist), attns_all)
            else:
                logits_all = torch.stack(logits_all, dim=1)
               
                res = (logits_all, attns_all)

            return res, angle[:,self.time_horizon:], distance[:,self.time_horizon:]

        res, probs = self(image_array, angle, distance, vego, seq_key)
        return res, angle, distance

    def validation_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, seq_key, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        res, probs = self(image_array, angle, distance, vego, seq_key)
    
        loss = self.calculate_loss(res, angle, distance)

        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)

            self.log_dict({"val_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"val_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        self.log_dict({"val_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        _, image_array, vego, angle, distance, seq_key, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        if self.time_horizon > 1:
            print(f"DEBUG: time_horizon={self.time_horizon}")
            logits_all = []
            logits_angle_all = []
            logits_distance_all = []
            angle_autoreg = angle.clone()
            attns_all = []
            distance_autoreg = distance.clone()
            for i in range(self.time_horizon, vego.shape[1], self.time_horizon):
                for j in range(self.time_horizon):
                    input_ids_img, input_ids_vego, input_ids_angle, input_ids_distance = image_array[:,0:i+j, :, :, :], vego[:,0:i+j], angle_autoreg[:,0:i+j], distance_autoreg[:,0:i+j]
                    if self.multitask == "angle" and len(logits_all) > 0:
                        
                        angle_autoreg[:, i+j] = logits_all[-1].squeeze(-1)
                    if self.multitask == "distance" and len(logits_all) > 0:
                        
                        distance_autoreg[:, i+j] = logits_all[-1].squeeze(-1)
                    if self.multitask == "multitask":
                        res, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego, seq_key)
                        logits, attns = res
                        param_angle, param_dist= logits[2], logits[3]
                        logits_angle, logits_distance = logits[0][:, -1], logits[1][:, -1]
                        angle_autoreg[:, i+j] = logits_angle.squeeze(-1)
                        distance_autoreg[:, i+j] = logits_distance.squeeze(-1)
                        logits_angle_all.append(logits_angle)
                        logits_distance_all.append(logits_distance)
                        
                       
                    else:
                        res, probs = self(input_ids_img, input_ids_angle, input_ids_distance, input_ids_vego, seq_key)
                        logits, attns = res
                        logits = logits[:, -1]
                        logits_all.append(logits)
                    attns_all.append(attns if attns is not None else torch.zeros_like(attns))

            if self.multitask == "multitask":
                logits_angle_all = torch.stack(logits_angle_all, dim=1)
                logits_distance_all = torch.stack(logits_distance_all, dim=1)
                loss = self.calculate_loss(((logits_angle_all, logits_distance_all, param_angle, param_dist), attns_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:])
                loss_angle, loss_dist, param_angle, param_dist = loss
                param_angle, param_dist = 0.3, 0.7
                loss = (param_angle * loss_angle) + (param_dist * loss_dist)
                loss_mae = self.calculate_mae_loss(((logits_angle_all, logits_distance_all, param_angle, param_dist), attns_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:])
                loss_angle_mae, loss_distance_mae, param_angle, param_dist = loss_mae
                loss_mae = (param_angle * loss_angle_mae) + (param_dist * loss_distance_mae)
                self.log("test_loss_dist_mae", loss_distance_mae, on_epoch=True, batch_size=self.bs, sync_dist=True)
                self.log("test_loss_angle_mae", loss_angle_mae, on_epoch=True, batch_size=self.bs, sync_dist=True)
                self.log("test_loss_mae", loss_mae, on_epoch=True, batch_size=self.bs, sync_dist=True)
                self.log("test_loss_angle", loss_angle, on_epoch=True, batch_size=self.bs, sync_dist=True)
                self.log("test_loss_distance", loss_dist, on_epoch=True, batch_size=self.bs, sync_dist=True)
                self.log("test_loss", loss, on_epoch=True, batch_size=self.bs, sync_dist=True)
                return (loss_angle, loss_dist)
            else:
                logits_all = torch.stack(logits_all, dim=1)
                loss = self.calculate_loss((logits_all, attns_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:])
                self.log("test_loss", loss, on_epoch=True, batch_size=self.bs, sync_dist=True)   
                loss_mae = self.calculate_mae_loss((logits_all, attns_all), angle[:,self.time_horizon:], distance[:,self.time_horizon:]) 
                self.log("test_loss_mae", loss_mae, on_epoch=True, batch_size=self.bs, sync_dist=True)
                return loss

        _, image_array, vego, angle, distance, seq_key, m_lens, i_lens, s_lens, a_lens, d_lens = batch
        res, probs = self(image_array, angle, distance, vego, seq_key)
        loss = self.calculate_loss(res, angle, distance)
        loss_mae = self.calculate_mae_loss(res, angle, distance)
        if self.multitask == "multitask":
            loss_angle, loss_dist, param_angle, param_dist = loss
            param_angle, param_dist = 0.3, 0.7
            loss = (param_angle * loss_angle) + (param_dist * loss_dist)
            self.log_dict({"test_loss_dist": loss_dist}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"test_loss_angle": loss_angle}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            loss_angle_mae, loss_distance_mae, param_angle, param_dist = loss_mae
            self.log_dict({"test_loss_angle_mae": loss_angle_mae}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            self.log_dict({"test_loss_distance_mae": loss_distance_mae}, on_epoch=True, batch_size=self.bs, sync_dist=True)
            return (loss_angle, loss_dist)
        self.log_dict({"test_loss": loss}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        self.log_dict({"test_loss_mae": loss_mae}, on_epoch=True, batch_size=self.bs, sync_dist=True)
        return loss
    #------------------------------------------------------------

    def train_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log_dict({"train_loss_accumulated": losses }, batch_size=self.bs, sync_dist=True)

    def validation_epoch_end(self, outputs):
        losses = torch.mean(torch.stack([x for x in outputs]))
        self.log_dict({"val_loss_accumulated": losses }, batch_size=self.bs, sync_dist=True)

    def test_epoch_end(self, outputs):
        if self.multitask == "multitask":
            loss_angles = torch.stack([x[0] for x in outputs])
            loss_dists  = torch.stack([x[1] for x in outputs])
            self.log_dict({
                "test_loss_angle_accum": loss_angles.mean(),
                "test_loss_distance_accum": loss_dists.mean()
            }, batch_size=self.bs, sync_dist=True)
        else:
            # single task
            losses = torch.stack(outputs)
            self.log("test_loss_accumulated", losses.mean(), batch_size=self.bs, sync_dist=True)
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
        g_opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        return g_opt

    def get_dataloader(self, dataset_type):
        ds = CommaDataset(dataset_type=dataset_type, multitask=self.multitask if not self.intervention else "intervention", ground_truth=self.ground_truth, dataset_path=self.dataset_path, dataset_fraction=self.dataset_fraction)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
