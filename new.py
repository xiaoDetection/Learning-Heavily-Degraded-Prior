import torch
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import numpy as np
import cv2
import os

def udcp(img_np):
    img = torch.from_numpy(img_np).float().cuda().permute(2, 0, 1).unsqueeze(0) # (b, c, h, w)
    min_channel = torch.min(img, dim=1, keepdim=True).values
    dark = -F.max_pool2d(-min_channel, 9, 1, 4)

    w = dark.size(-1)
    ind = dark.argmax()
    a = ind % w
    b = ind // w
    atomspheric_light = img[..., b, a]

    im = img / atomspheric_light.reshape(1, 3, 1, 1)
    img_gray = torch.min(im, 1, keepdim=True).values
    d = -F.max_pool2d(-img_gray, 9, 1, 4)
    trans = torch.clamp((1 - d), 0.1, 0.9).expand(-1, 3, -1, -1)
    
    filter = GuidedFilter(50, 1e-3)
    mask = torch.clamp(filter(img / 255.0, trans), 0.1, 0.9)
    mask = mask.mean(1, keepdim=True).squeeze(0).permute(1, 2, 0).cpu().numpy()

    mask[mask > 0.5] = 0
    mask[mask > 0] = 1

    return img_np * mask

input = '/temp3/xiaojiewen/heavily-degraded-prior/data/dfui/images'
output = '/temp3/xiaojiewen/heavily-degraded-prior/data/dfui/temp'

imgs = os.listdir(input)
imgs = sorted(imgs)
for img in imgs:
    if not img.endswith('jpg'):
        break

    print(img)
    img_dir = os.path.join(input, img)
    img_cv = cv2.imread(img_dir)
    cv2.imwrite(os.path.join(output, img), udcp(img_cv))
