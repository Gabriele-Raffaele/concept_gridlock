'''
Reader of the .pt files
'''

import torch

# Choose which batch number to load (from 0 to 98)

# Path template
path = (
   "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/kaggle/working/road-save/comma/cache/resnet50I3D512-Pkinetics-b4s8x1x1-commat3-h3x3x3/batch_concepts-30-08/b0c9d2329ad1606b_2018-07-27--06-03-57_11.pt"
)
# Fill in the batch number


# Load the file
data = torch.load(path, map_location="cpu", weights_only=True)

# Inspect keys
print("Available keys:", data.keys())


concepts = data["textual_concepts"]

with open('/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/concept_gridlock/scenarios/road_concepts.txt', 'w') as f:
    for concept in concepts:
        f.write(concept + '\n')

print("Road concepts have been written to 'road_concepts.txt'.")
