#requires pytorch_lightning==1.9.4
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
import json
import glob
import re
#function to save predictions
def save_preds(logits, target, save_name, p, time_horizon=1):
    logits_squeezed = logits.squeeze(-1)  # toglie solo l'ultima dim, shape: [B, S]
    if time_horizon > 1:
        target = target[:, time_horizon-1::time_horizon]
        logits_squeezed = logits_squeezed[:, time_horizon-1::time_horizon]

    b, s = target.shape
    try:
        logits_reshaped = logits_squeezed.reshape(b*s)
    except Exception as e:
        print(f"Error reshaping logits: {e}")
        raise e
    df = pd.DataFrame()
    df['logits'] = logits_reshaped.tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    write_header = not os.path.exists(f'{p}/{save_name}.csv')
    df.to_csv(f'{p}/{save_name}.csv', mode='a', index=False, header=write_header)
    print(f"Saved predictions to {p}/{save_name}.csv")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="multitask", type=str)  
    parser.add_argument('-train', action=argparse.BooleanOptionalAction)  
    parser.add_argument('-test', action=argparse.BooleanOptionalAction)
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-train_concepts', default=False, action=argparse.BooleanOptionalAction) 
    parser.add_argument('-n_scenarios', default=100, type=int) 
    parser.add_argument('-scenario_type', default="not_specified", type=str) 
    parser.add_argument('-dataset_fraction', default=1, type=float) 
    parser.add_argument('-dataset', default="comma", type=str)  
    parser.add_argument('-backbone', default="none", type=str) 
    parser.add_argument('-dataset_path', default='/kaggle/input/final-hdf5-files', type=str)
    parser.add_argument('-concept_features', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-new_version', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-intervention_prediction', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-save_path', default="/kaggle/working/", type=str) 
    parser.add_argument('-max_epochs', default=1, type=int) 
    parser.add_argument('-bs', default=1, type=int) 
    parser.add_argument('-ground_truth', default="normal", type=str) 
    parser.add_argument('-dev_run', default=False, type=bool) 
    parser.add_argument('-checkpoint_path', default='', type=str)
    parser.add_argument('-time_horizon', default=1, type=int, help="Time horizon for predictions")
    parser.add_argument('-concept_source', default="retinanet", type=str,choices=["clip", "retinanet"], help="Source of concept logits: clip or retinanet")
    parser.add_argument('-seed', default=42, type=int)
    return parser


def main():    
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 0 and torch.cuda.get_device_capability()[0] >= 7:
        # Set the float32 matrix multiplication precision to 'high'
        torch.set_float32_matmul_precision('high')
    
    parser = get_arg_parser()
    args = parser.parse_args()
    task = args.task
    
    if args.concept_source == "clip":
        args.n_scenarios = 100
    else:  # retinanet
        args.n_scenarios = 148
    
    pl.seed_everything(args.seed, workers=True)
    print(f"TASK = {args.task}, CONCEPT_FEATURES = {args.concept_features}, CONCEPT_SOURCE = {args.concept_source}, DATASET_PATH = {args.dataset_path}, DATASET_FRACTION = {args.dataset_fraction}, BACKBONE = {args.backbone}, TRAIN_CONCEPTS = {args.train_concepts}, TIME_HORIZON = {args.time_horizon}  ")
    #EarlyStopping: stop training when val_loss_accumulated does not improve
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", 
                                        min_delta=0.05, 
                                        patience=5, 
                                        verbose=False, 
                                        mode="max")
    
    
    model = VTN(multitask=task, 
                backbone=args.backbone, 
                concept_features=args.concept_features, 
                device = f"cuda:{args.gpu_num}", 
                train_concepts=args.train_concepts,
                concept_source=args.concept_source)

    #Wrapper that links the model to data, loss and optimizer
    module = LaneModule(model, 
                        multitask=task, 
                        dataset = args.dataset, 
                        bs=args.bs, 
                        ground_truth=args.ground_truth, 
                        intervention=args.intervention_prediction, 
                        dataset_path=args.dataset_path, 
                        dataset_fraction=args.dataset_fraction,
                        time_horizon=args.time_horizon
                        )
    #Set where to save the checkpoints, when to save them with the ModelCheckpoint callback, and the logger for TensorBoard
    ckpt_pth = f"/kaggle/working/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}_{args.concept_source}"
    path = ckpt_pth + "/lightning_logs/" 
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_callback = ModelCheckpoint(save_top_k=2, 
                                            monitor="val_loss_accumulated",
                                            mode = "min",
                                            save_last=True,
                                            filename="{epoch}-{val_loss_accumulated:.4f}"
                                            
                                            )

    vs = os.listdir(path)
    filt = [elem for elem in vs if 'version' in elem]

    if not args.new_version:
        if filt:
            versions = sorted([int(elem.split('_')[-1]) for elem in filt])
            version = versions[-1]
        else:
            version = None
    else:
        version = None

    logger = TensorBoardLogger(save_dir=ckpt_pth, version=version)

   
    if version is not None:
        resume_path = os.path.join(path, f"version_{version}", "checkpoints")
        if os.path.exists(resume_path):
            files = [f for f in os.listdir(resume_path) if f.endswith(".ckpt") and f.startswith("epoch")]
            print(f"Found existing version: {version}, files: {files}")

            if files:
                
                def get_loss_from_name(fname):
                    match = re.search(r"val_loss_accumulated[=]?(\d+\.\d+)", fname)
                    return float(match.group(1)) if match else float("inf")

                best_ckpt = min(files, key=get_loss_from_name)
                resume = os.path.join(resume_path, best_ckpt)
            else:
                # fallback a last.ckpt
                last_ckpt = os.path.join(resume_path, "last.ckpt")
                resume = last_ckpt if os.path.exists(last_ckpt) else None
        else:
            resume = None
    else:
        resume = None

    print(f"RESUME FROM: {resume}")

#------------------------------------------------------------
    #training setup
    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        #gpus=2,
        accelerator='gpu',
        devices=2 if torch.cuda.is_available() else None,  
        strategy="ddp",
        logger=logger,
      
        max_epochs=args.max_epochs,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), 
                   checkpoint_callback 
                   #early_stop_callback
                   ],
        log_every_n_steps=1,
        )
    test_trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        #gpus=2,
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,  
        strategy="auto",    
        logger=logger,
        
        max_epochs=args.max_epochs,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
        log_every_n_steps=1,
        )
    
    #start training and saves args in a yaml file
    if args.train:
        if args.checkpoint_path != '':
            resume = args.checkpoint_path
        trainer.fit(module, ckpt_path=resume)
        save_path = "/".join(checkpoint_callback.best_model_path.split("/")[:-1])
        print(f'saving hparams at {save_path}')
        with open(f'{save_path}/hparams.yaml', 'w') as f:
            yaml.dump(args, f)
    

    #Use the specified checkpoint to do predictions         
    if args.test:
        #if train and test are not computed together then checkpoint_callback.best_model_path will not be set because the model was not trained
        #ckpt_path = args.checkpoint_path if args.checkpoint_path != '' else checkpoint_callback.best_model_path
        #Build checkpoint path -G.R.
        if args.checkpoint_path == '':
            ckpt_path = resume
            if ckpt_path is None:
                ckpt_root = f"/kaggle/working/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}_{args.concept_source}"
                #find the latest version -G.R.
                versions = glob.glob(os.path.join(ckpt_root, "lightning_logs", "version_*"))
                if not versions:
                    raise FileNotFoundError("None found")
                latest_version = max(versions, key=os.path.getmtime)

                # Find checkpoints in the latest version -G.R.
                ckpt_files = glob.glob(os.path.join(latest_version, "checkpoints", "*.ckpt"))
                if not ckpt_files:
                    raise FileNotFoundError(f"Checkpoint file not found")
                ckpt_path = max(ckpt_files, key=os.path.getmtime)
        else:
            ckpt_path = args.checkpoint_path
        print(f"Using checkpoint: {ckpt_path}")
        test_results = test_trainer.test(module, ckpt_path=ckpt_path)
        result_dir = os.path.dirname(ckpt_path)
        with open(f"{result_dir}/test_metrics.json", "w") as f:
            json.dump(test_results, f, indent=4)
        
        preds = test_trainer.predict(module, ckpt_path=None)
        #save_path =  "."
        for pred in preds:
            if args.task != "multitask":
                res, preds_1, preds_2 = pred[0], pred[1], pred[2]
                
                prediction, attention = res
                if args.task == "angle":
                    save_preds(prediction, preds_1, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}_{args.concept_source}", "/kaggle/working", module.time_horizon)
                else:
                    save_preds(prediction, preds_2, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}_{args.concept_source}", "/kaggle/working", module.time_horizon)

            else:
                res, angle, dist = pred[0], pred[1], pred[2]
                (preds_angle, preds_dist, param_angle, param_dist), attention = res
                save_preds(preds_angle, angle, f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.concept_source}", "/kaggle/working", module.time_horizon)
                save_preds(preds_dist, dist, f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.concept_source}", "/kaggle/working", module.time_horizon)

if __name__ == "__main__":
    main()
