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
#function to save predictions
def save_preds(logits, target, save_name, p):
    b, s = target.shape
    df = pd.DataFrame()
    df['logits'] = logits.squeeze().reshape(b*s).tolist()
    df['target'] = target.squeeze().reshape(b*s).tolist()
    df.to_csv(f'{p}/{save_name}.csv', mode='a', index=False, header=False)
    print(f"Saved predictions to {p}/{save_name}.csv")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', default="multitask", type=str)  
    parser.add_argument('-train', action=argparse.BooleanOptionalAction)  
    parser.add_argument('-test', action=argparse.BooleanOptionalAction)
    parser.add_argument('-gpu_num', default=0, type=int) 
    parser.add_argument('-train_concepts', action=argparse.BooleanOptionalAction) 
    parser.add_argument('-n_scenarios', default=100, type=int) 
    parser.add_argument('-scenario_type', default="not_specified", type=str) 
    parser.add_argument('-dataset_fraction', default=1, type=float) 
    parser.add_argument('-dataset', default="comma", type=str)  
    parser.add_argument('-backbone', default="none", type=str) 
    parser.add_argument('-dataset_path', default='/kaggle/input/filtered-chunk-hdf5', type=str)
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
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"

    if torch.cuda.device_count() > 0 and torch.cuda.get_device_capability()[0] >= 7:
        # Set the float32 matrix multiplication precision to 'high'
        torch.set_float32_matmul_precision('high')
    
    
    
    parser = get_arg_parser()
    args = parser.parse_args()
    task = args.task
    #print per debugging
    print(f"TASK = {args.task}, CONCEPT_FEATURES = {args.concept_features}")
    #EarlyStopping: stop training when val_loss_accumulated does not improve
    early_stop_callback = EarlyStopping(monitor="val_loss_accumulated", 
                                        min_delta=0.05, 
                                        patience=5, 
                                        verbose=False, 
                                        mode="max")
    
    #capire come impostare più gpu - G.R.
    model = VTN(multitask=task, 
                backbone=args.backbone, 
                concept_features=args.concept_features, 
                device = f"cuda:{args.gpu_num}", 
                train_concepts=args.train_concepts)
    
    ''' if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda")
        model = DataParallel(model, device_ids=[0,1])  # scegli 0,1,2… in base alle GPU
        model.to(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        (peppe )
    '''
    #Wrapper that links the model to data, loss and optimizer
    module = LaneModule(model, 
                        multitask=task, 
                        dataset = args.dataset, 
                        bs=args.bs, 
                        ground_truth=args.ground_truth, 
                        intervention=args.intervention_prediction, 
                        dataset_path=args.dataset_path, 
                        dataset_fraction=args.dataset_fraction)
    #set where to save the checkpoints, when to save them with the ModelCheckpoint callback, and the logger for TensorBoard
    ckpt_pth = f"/kaggle/working/ckpts_final_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.dataset_fraction}"
    checkpoint_callback = ModelCheckpoint( save_top_k=2, 
                                          #dirpath=ckpt_pth, (peppe)
                                           #filename='best-{epoch}-{val_loss:.2f}', (peppe)
                                           #mode="min", (peppe)
                                        monitor="val_loss_accumulated")
    
    logger = TensorBoardLogger(save_dir=ckpt_pth)
#------------------------------------------------------------
    # Check if the checkpoint path exists, then it sets its for resuming training.
    # Tutta questa parte da peppe non c'è
    path = ckpt_pth + "/lightning_logs/" 
    if not os.path.exists(path):
        os.makedirs(path)
    vs = os.listdir(path)
    filt = []
    
    f_name, resume_path = 'None', 'None'
    if not args.new_version and not args.test:
        for elem1 in vs: 
            if 'version' in elem1:
                filt.append(elem1)
        #Ho aggiunto l'if filt -G.R.
        if filt: 
            versions =[elem.split("_")[-1]for elem in filt]
            versions = sorted(versions)
            version = f"version_{versions[-1]}"
            resume_path = path + version + "/checkpoints/"
            files = os.listdir(resume_path)
            print(files)
            for f in files: 
                if "ckpt" in f:
                    f_name = f
                    break
                else: 
                    f_name = None
            print(f_name)
        else:
            print("⚠️ No previous versions found — skipping resume.")
            resume_path, f_name = None, None
    #ho leggermente modificato questa parte per evitare errori in caso di resum_path che f_name None -G.R.
    # original code: resume = None if args.new_version or args.test and f_name != None else resume_path + f_name
    resume = None if (args.new_version or args.test or f_name is None or resume_path is None) else resume_path + f_name
    print(f"RESUME FROM: {resume}")

#------------------------------------------------------------
    #training setup
    trainer = pl.Trainer(
        fast_dev_run=args.dev_run,
        #gpus=2,
        accelerator='gpu',
        devices= 1 if torch.cuda.is_available() else None, 
        logger=logger,
        resume_from_checkpoint= resume,
        max_epochs=args.max_epochs,
        default_root_dir=ckpt_pth ,
        callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
        #, EarlyStopping(monitor="train_loss", mode="min")],#in case we want early stopping
        log_every_n_steps=10,
        )
    save_path = args.save_path
    #start training and saves args in a yaml file
    if args.train:
        trainer.fit(module) # ho aggiunto il resume qui, originalmente era nel trainer. original code: trainer.fit(module) - G.R.
        save_path = "/".join(checkpoint_callback.best_model_path.split("/")[:-1])
        print(f'saving hparams at {save_path}')
        with open(f'{save_path}/hparams.yaml', 'w') as f:
            yaml.dump(args, f)


    #Use the specified checkpoint to do predictions         
    ckpt_path=args.checkpoint_path
    p = "/".join(ckpt_path.split("/")[:-2])
    #preds = trainer.test(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    #best_ckpt = checkpoint_callback.best_model_path (peppe)
    # ???
    preds = trainer.predict(module, ckpt_path=ckpt_path if ckpt_path != '' else "best")
    #save_path =  "."
    #TODO: Considerare solo il multitask
    for pred in preds:
        if args.task != "multitask":
            predictions, preds_1, preds_2 = pred[0], pred[1], pred[2] 
            save_preds(predictions, preds_1, f"{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}_{args.n_scenarios}", save_path)
        else:
            res, angle, dist = pred[0], pred[1], pred[2]
            (preds_angle, preds_dist, param_angle, param_dist), attention = res
            save_preds(preds_angle, angle, f"angle_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)
            save_preds(preds_dist, dist, f"dist_multi_{args.dataset}_{args.task}_{args.backbone}_{args.concept_features}", save_path)