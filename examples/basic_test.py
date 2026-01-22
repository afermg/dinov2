import numpy
import torch

# device = torch.cuda.device("cuda:0")

# DINOv2
device = torch.device(3)
dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitl14",  # , pretrained=False
).to(device)

tile_size = 224
data = numpy.random.random_sample((1, 3, tile_size, tile_size))
torch_tensor = torch.from_numpy(data).float().to(device)

tmp = dinov2(torch_tensor)
print(dinov2.blocks[0].attn.qkv.weight[0, :2].detach().cpu().numpy())
# pretrained: [-0.00366281, 0.00109316]
# not pretrained: [0.00304486 0.02964859]
print(tmp.shape)
