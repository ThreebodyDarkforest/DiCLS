from model.backbone import ImageEncoder
from model.utils.configs import Config
import torch

cfg = Config()
test = ImageEncoder(cfg.encoder.visual, 'swint_v1', frozen_stages=7)
x = torch.randn((1, 3, 224, 224))
x = test(x)

for k, v in x.items():
    if k != "all_hidden":
        print(k, v.shape)
    else:
        print(k, [a.shape for a in v])