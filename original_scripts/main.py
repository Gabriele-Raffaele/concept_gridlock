import pytorch_lightning as pl
from model import *
from module import * 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch 
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
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
    parser.add_argument('-task', default="", type=str)  
    parser.add_argument('-train', action=argparse.BooleanOptionalAction)  
    parser.add_argument('-test', action=argparse.BooleanOptionalAction)
    
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-train_concepts', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-n_scenarios', default=643, type=int) 
    parser.add_argument('-scenario_type', default="not_specified", type=str) 
    
    parser.add_argument('-dataset_fraction', default=1, type=float) 
    parser.add_argument('-dataset', default="comma", type=str)  
    parser.add_argument('-backbone', default="resnet", type=str) 
    parser.add_argument('-dataset_path', default="/kaggle/input/hdf5-chunk1/", type=str) 
    parser.add_argument('-concept_features', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-new_version', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-intervention_prediction', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-save_path', default="", type=str) 
    parser.add_argument('-max_epochs', default=1, type=int) 
    parser.add_argument('-bs', default=1, type=int) 
    parser.add_argument('-ground_truth', default="normal", type=str) 
    parser.add_argument('-dev_run', default=False, type=bool) 
    parser.add_argument('-checkpoint_path', default='', type=str)
    return parser

if __name__ == "__main__":
   # empty the cache and set the max split size to avoid fragmentation 
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"

    if torch.cuda.device_count() > 0 and torch.cuda.get_device_capability()[0] >= 7:
        # Set the float32 matrix multiplication precision to 'high'
        torch.set_float32_matmul_precision('high')
    
    # parse the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    multitask = args.task
   
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", min_delta=0.05, patience=5, verbose=False, mode="max")
    #initialize the model and module, the module is a wrapper around the model that handles the training and testing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = VTN(multitask=multitask, backbone=args.backbone, concept_features=args.concept_features, device = device, train_concepts=args.train_concepts)
    module = LaneModule(model, multitask=multitask, dataset = args.dataset, bs=args.bs, ground_truth=args.ground_truth, intervention=args.intervention_prediction, dataset_path=args.dataset_path, dataset_fraction=args.dataset_fraction)

    ckpt_pth = f"/kaggle/working/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}"    
    print(ckpt_pth)
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss_accumulated")
    logger = TensorBoardLogger(save_dir=ckpt_pth)

    path = ckpt_pth + "/lightning_logs/" 
    if not os.path.exists(path):
        os.makedirs(path)
    vs = os.listdir(path)
    filt = [elem for elem in vs if 'version' in elem]
    f_name, resume_path = 'None', 'None'
    resume = None

    if not args.new_version and not args.test:
        if filt:
            versions = [elem.split("_")[-1] for elem in filt]
            versions = sorted(versions)
            version = f"version_{versions[-1]}"
            resume_path = os.path.join(path, version, "checkpoints")
            if os.path.exists(resume_path):
                files = os.listdir(resume_path)
                for f in files:
                    if "ckpt" in f:
                        f_name = f
                        break
                if f_name and os.path.exists(os.path.join(resume_path, f_name)):
                    resume = os.path.join(resume_path, f_name)
        else:
            print("⚠️ No resume.")
    
    print(f"RESUME FROM: {resume}")

    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        #gpus=2, 
        accelerator='gpu',
        devices=[args.gpu_num] if torch.cuda.is_available() else None, 
        logger=logger,
        resume_from_checkpoint= resume, 
        max_epochs=args.max_epochs,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
        #, EarlyStopping(monitor="train_loss", mode="min")],#in case we want early stopping
        )
    #save_path = args.save_path
    #in modo che salva sulla working directory di kaggle
    save_path = ckpt_pth
    if args.train:
        trainer.fit(module)
        save_path = "/".join(checkpoint_callback.best_model_path.split("/")[:-1])
        print(f'saving hparams at {save_path}')
        with open(f'{save_path}/hparams.yaml', 'w') as f:
            yaml.dump(args, f)
    ckpt_path=args.checkpoint_path
    p = "/".join(ckpt_path.split("/")[:-2])
    #preds = trainer.test(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    preds = trainer.predict(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    #save_path =  "."
    for pred in preds:
        if args.task != "multitask":
            predictions, preds_1, preds_2 = pred[0], pred[1], pred[2] 
            save_preds(predictions, preds_1, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}", save_path)
        else:
            preds, angle, dist = pred[0], pred[1], pred[2]
            preds_angle, preds_dist = preds[0], preds[1]
            save_preds(preds_angle, angle, f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)
            save_preds(preds_dist, dist, f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)