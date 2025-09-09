import torch


ckpt_path = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/version_0/checkpoints/epoch=92-val_loss_accumulated=1.1591.ckpt"


ckpt = torch.load(ckpt_path, map_location="cpu")


print("Principal keys in checkpoint:", ckpt.keys())



if 'hyper_parameters' in ckpt:
    print("\nHyperparameters saved in checkpoint:")
    for k, v in ckpt['hyper_parameters'].items():
        print(f"{k}: {v}")


if 'state_dict' in ckpt:
    print("\nExample of layers in the model:")
    for i, k in enumerate(ckpt['state_dict'].keys()):
        print(k)