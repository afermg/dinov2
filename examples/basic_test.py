import numpy as np
import torch

device = torch.cuda.device("cuda:0")

# DINOv2
dinov2_vits14_lc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").to(
    "cuda:0"
)

random_3d_array = np.random.rand(1, 3, 420, 420)
torch_tensor = torch.from_numpy(random_3d_array).float().cuda()

dinov2_vits14_lc(torch_tensor)
