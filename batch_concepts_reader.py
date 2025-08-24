'''
Reader of the .pt files
'''

import torch

# Choose which batch number to load (from 0 to 98)
batch_num = 80

# Path template
path_template = (
    "kaggle/working/road-save/comma/cache/"
    "resnet50I3D512-Pkinetics-b4s8x1x1-commat3-h3x3x3/"
    "batch_concepts-30-08/batch_{batch_num:06d}_concepts.pt"
)
# Fill in the batch number
path = path_template.format(batch_num=batch_num)

# Load the file
data = torch.load(path, map_location="cpu", weights_only=True)

# Inspect keys
print("Available keys:", data.keys())

# Access values
print("Textual Concepts:", data["concepts"])
print("Batch size:", data["batch_size"])
print("Sequence length:", data["seq_len"])
print("Num concepts:", data["num_concepts"])
print("Batch index:", data["batch_idx"])

print("Unique videos:", data["unique_videos"])

# Concepts shapes
print("Concept logits shape:", data["concepts"].shape) #[batch_size, seq_len, num_concepts]
#IMPORTANT: Concepts are from 1 to num_concepts, the first value is a confidence score.

# First video info entry
if data["video_info"]:
    print("First video info:", data["video_info"][0])



