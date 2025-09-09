'''
Reader of the .pt files
'''

import torch

# Choose which batch number to load (from 0 to 98)

# Path template
path = (
   "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/Similarity/concepts_with_noise/b0c9d2329ad1606b_2018-08-17--14-55-39_4.pt"
)
# Fill in the batch number


# Load the file
data = torch.load(path, map_location="cpu", weights_only=True)

# Inspect keys
print("Available keys:", data.keys())


concepts = data["concepts"]
#print concepts at frame 152 (should be all zeros)
print("concepts at frame 152:", concepts[211])  # (num_concepts
