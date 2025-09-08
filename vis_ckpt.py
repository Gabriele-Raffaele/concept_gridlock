import torch

# Percorso al checkpoint
ckpt_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/version_0/checkpoints/epoch=92-val_loss_accumulated=1.1591.ckpt"

# Carica il checkpoint come dizionario
ckpt = torch.load(ckpt_path, map_location="cpu")

# Mostra le chiavi principali
print("Chiavi principali nel checkpoint:", ckpt.keys())

# Spesso trovi:
# 'state_dict' -> i pesi del modello
# 'hyper_parameters' -> gli hparams salvati
# 'optimizer_states', 'lr_schedulers', 'epoch', ecc.

# Controlla gli hyperparameters
if 'hyper_parameters' in ckpt:
    print("\nHyperparameters salvati nel checkpoint:")
    for k, v in ckpt['hyper_parameters'].items():
        print(f"{k}: {v}")

# Controlla lo state_dict (nomi dei layer)
if 'state_dict' in ckpt:
    print("\nEsempio di layer nel modello:")
    for i, k in enumerate(ckpt['state_dict'].keys()):
        print(k)