import tempfile
import os
import os.path as osp
import torch
import torch.nn.functional as F
import random
import multiprocessing

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from mmcv.utils import build_from_cfg
from .builder import PIPELINES, build_dataset
from guided_filter_pytorch.guided_filter import GuidedFilter

# UDCP
# P. D. Jr., E. R. do Nascimento, F. Moraes, S. S. C. Botelho, and M. F. M.
# Campos, “Transmission estimation in underwater single images,” in
# ICCV Workshops, 2013, pp. 825–830.
def udcp(img_np, filter, thr=0.5):
    img = torch.from_numpy(img_np).float().cuda().permute(2, 0, 1).unsqueeze(0) # (B, 3, H, W)
    min_channel = torch.min(img, dim=1, keepdim=True).values
    dark = -F.max_pool2d((-min_channel), 9, 1, 4)

    w = dark.size(-1)
    ind = dark.argmax()
    a = ind % w
    b = torch.div(ind, w, rounding_mode='floor')
    atomspheric_light = img[..., b, a]

    im = img / atomspheric_light.reshape(1, 3, 1, 1)
    img_gray = torch.min(im, 1, keepdim=True).values
    d = -F.max_pool2d(-(img_gray), 9, 1, 4)
    trans = torch.clamp((1 - d), 0.1, 0.9).expand(-1, 3, -1, -1)

    mask = torch.clamp(filter(img / 255.0, trans), 0.1, 0.9)
    mask = mask.mean(1, keepdim=True).squeeze(0).permute(1, 2, 0).cpu().numpy() # (H, W, 1)

    mask[mask > thr] = 0
    mask[mask > 0] = 1
    return img_np * mask

class ImgLoader:
    def __init__(self,
                img_prefix,
                shuffle=True):
                
        assert os.path.exists(img_prefix), '%s not exists'%img_prefix
        self.img_prefix = img_prefix
        self.load_img = build_from_cfg(dict(type='LoadImageFromFile'), PIPELINES)
        self.names = sorted([n for n in os.listdir(self.img_prefix) if n.endswith('.jpg')])
        self.length = len(self.names)
        self.shuffle = shuffle
        self.glob_idx = multiprocessing.Value('i', 0)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.glob_idx.get_lock():
            name = self.names[self.glob_idx.value]
            self.glob_idx.value += 1
            if self.glob_idx.value % self.length == 0:
                self.glob_idx.value = 0
                if self.shuffle:
                    random.shuffle(self.names)
        return self.get_img_by_name(name)


    def get_img_by_name(self, filename):
        result = dict(
            img_prefix=self.img_prefix,
            img_info={'filename':filename}
        )
        result = self.load_img(result)
        return result


@DATASETS.register_module()
class CocoDatasetCustom:
    '''
    Load underwater image, HD_u and HD_f at the same time

    Args:
        dataset (dict): config of coco dataset for 
        pipeline (list[dict]): Processing pipeline.
        img_dfui_prefix (str, optional): DFUI_H image prefiex
        mask_thr (float, optional): mask threshold
        test_mode (bool, optional): If set True, annotation will not be loaded.
    '''
    def __init__(self,
                 dataset,
                 pipeline,
                 img_dfui_prefix=None,
                 mask_thr=0.5,
                 test_mode=False):
        dataset.test_mode = test_mode
        self.dataset = build_dataset(dataset)
        self.dataset.test_mode = test_mode
        self.CLASSES = self.dataset.CLASSES
        self.PALETTE = getattr(dataset, 'PALETTE', None)
        if not test_mode:
            self.flag = self.dataset.flag

        # self.img_masked_loader = ImgLoader(img_masked_prefix)
        self.mask_thr = mask_thr
        self.img_dfui_loader = ImgLoader(img_dfui_prefix) if img_dfui_prefix is not None else None
        self.filter = GuidedFilter(50, 1e-3)

        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        result = self.dataset[idx]
        result['img_fields'] = ['img']

        # load DFUI_H
        if self.img_dfui_loader is not None:
            result['img_dfui'] = udcp(self.img_dfui_loader[idx]['img'], self.filter, self.mask_thr)
            result['img_fields'].append('img_dfui')
            
        # load the corresponding UI_H
        result['img_masked'] = udcp(result['img'], self.filter, self.mask_thr)
        result['img_fields'].append('img_masked')

        result = self.pipeline(result)
        
        return result
   
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        return self.dataset.evaluate(
            results,
            metric,
            logger,
            jsonfile_prefix,
            classwise,
            proposal_nums,
            iou_thrs,
            metric_items
        )