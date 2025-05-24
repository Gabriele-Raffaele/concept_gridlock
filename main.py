import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from torch.nn import DataParallel
from pytorch_lightning.callbacks import ModelCheckpoint
from  pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
import pandas as pd 
import os

''' Save predictions to csv file'''
def save_preds(logits, target, save_name, p):
    b, s = target.shape
    df = pd.DataFrame()
    df['logits'] = logits.squeeze().reshape(b*s).tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    df.to_csv(f'{p}/{save_name}.csv', mode='a', index=False, header=False)

'''Define the argument parser'''
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default='angle', type=str)
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-gpu_num', default=0, type=int)
    parser.add_argument('-dataset', default='comma', type=str)
    parser.add_argument('-dataset_path', default='/kaggle/input/filtered-chunk1', type=str)
    parser.add_argument('-bs', default=8, type=int)
    parser.add_argument('-max_epochs', default=10, type=int)
    parser.add_argument('-ground_truth', default='desired', type=str)
    parser.add_argument('-new_version', action='store_true')
    parser.add_argument('-intervention_prediction', action='store_true')
    parser.add_argument('-dev_run', action='store_true')
    parser.add_argument('-backbone', default='resnet', type=str, choices=['resnet', 'vit'], help='Backbone model type')
    parser.add_argument('-concept_features', action='store_true', help='Use concept features')
    parser.add_argument('-train_concepts', action='store_true', help='Train concept features')
    parser.add_argument('-dataset_fraction', default=1.0, type=float, help='Fraction of dataset to use')
    return parser
if __name__ == "__main__":
    

    args_list = [
        '-train',
        '-task', 'multitask',  # o 'angle'/'distance'
        '-dataset', 'comma',
        '-dataset_path', '/kaggle/input/filtered-chunk1',
        '-bs', '2',  # Batch 
        '-max_epochs', '1',
        '-backbone', 'resnet',
        '-ground_truth', 'desired',
        '-concept_features',  
        '-train_concepts'   
    ]
    
    parser = get_arg_parser()
    args = parser.parse_args(args=args_list)

    model = VTN(
        multitask=args.task,
        backbone=args.backbone,
        concept_features=args.concept_features,
        device="cpu",
        train_concepts=args.train_concepts
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda")
        model = DataParallel(model, device_ids=[0,1])  # scegli 0,1,2… in base alle GPU
        model.to(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    module = LaneModule(model, multitask=args.task, dataset = args.dataset, bs=args.bs, ground_truth=args.ground_truth, intervention=args.intervention_prediction, dataset_path=args.dataset_path, dataset_fraction=args.dataset_fraction)


    # Configurazione checkpoint e logger
    ckpt_pth = f"/kaggle/working/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_pth,
        filename='best-{epoch}-{val_loss:.2f}',
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger(save_dir=ckpt_pth)

    # Configurazione trainer
    trainer = pl.Trainer(
       accelerator="gpu",
       devices=1,                # usa 2 GPU
       precision="16-mixed",             # mixed precision FP16 per dimezzare l’uso di VRAM
       max_epochs=args.max_epochs,
       logger=logger,
       callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=3)
        ],
        enable_checkpointing=True
    )

    # Training
    trainer.fit(module)
    
    # Salvataggio configurazione
    save_path = "/".join(checkpoint_callback.best_model_path.split("/")[:-1])
    with open(f'{save_path}/hparams.yaml', 'w') as f:
        yaml.dump(vars(args), f)  # Nota: usiamo vars() per convertire Namespace in dict
    
    print(f"Training completato! Checkpoint salvato in: {checkpoint_callback.best_model_path}")
    
    best_ckpt = checkpoint_callback.best_model_path
    preds = trainer.predict(module, ckpt_path=best_ckpt if best_ckpt else "best")

    for (logits, angle_gt, dist_gt) in preds:
        if args.task != "multitask":
            # single-task
            save_preds(
                logits, angle_gt,
                f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}",
                save_path
            )
        else:
            # estrai solo i primi due (angle_preds, dist_preds)
            angle_preds, dist_preds = logits[0], logits[1]

            save_preds(
                angle_preds, angle_gt,
                f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}",
                save_path
            )
            save_preds(
                dist_preds, dist_gt,
                f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}",
                save_path
            )

    print(f"Prediction completate, CSV salvati in: {save_path}")
