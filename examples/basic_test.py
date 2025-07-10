import numpy
import torch

# device = torch.cuda.device("cuda:0")

# DINOv2
dinov2_vits14_lc = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_lc").to(
    "cuda:0"
)

data = numpy.random.random_sample((1, 3, 420, 420))
torch_tensor = torch.from_numpy(data).float().cuda()

tmp = dinov2_vits14_lc(torch_tensor)
