from mmcv.runner import HOOKS, Hook, load_checkpoint
from mmcv import Config

import torch
import torch.nn as nn

class Kl_Loss:
    def __init__(self):
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = nn.Flatten()
    
    def __call__(self, input, target):
       input = self.log_softmax(self.flatten(input))
       target = self.softmax(self.flatten(target))
       return self.kl_div_loss(input, target)

@HOOKS.register_module()
class OurHook(Hook):
    def __init__(
        self,
        cfg,        # str, config file path 
        cp,         # str, pretrained model checkpoint file path 
        loss_weight=1,
    ):
        self.loss_weight = loss_weight
        self.cp = cp
        self.cfg = Config.fromfile(cfg)
        super(OurHook, self).__init__()

       

    def before_run(self, runner):
        from mmdet.models import build_backbone

        self.model = runner.model.module
        for par in self.model.parameters():
            par.requires_grad = False

        # build the pretrained model
        runner.logger.info('build the pretrained model')
        self.pretrained_model = build_backbone(self.cfg.pretrained_model).cuda()

        runner.logger.info('load checkpoint from %s' %self.cp)
        state_dict = torch.load(self.cp)
        state_dict_new = {k.replace('backbone.', '') : v for k, v in state_dict.items() if 'backbone' in k}
        self.pretrained_model.load_state_dict(state_dict_new, strict=False)

        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        # build loss
        self.kl_loss = Kl_Loss()

    def before_train_epoch(self, runner):
        if runner.epoch < 6 or (runner.epoch >= 12 and runner.epoch %2 == 0):
            # Training RFTM
            runner.train_RFTM = True
            runner.logger.info('Training RFTM')

            # freeze parameters
            for par in self.model.parameters():
                par.requires_grad = False
            for name, par in self.model.backbone.named_parameters():
               if 'RFTM' in name:
                   par.requires_grad = True

        else:
            # Finetune
            runner.train_RFTM = False
            runner.logger.info('Finetune')

            # freeze parameters
            for par in self.model.parameters():
                par.requires_grad = False
            for name, par in self.model.named_parameters():
                if 'layer3' in name or 'layer4' in name or 'backbone' not in name:
                    par.requires_grad = True

    def after_train_iter(self, runner):
        if runner.train_RFTM:
            self.DFUI_H1 = self.pretrained_model.forward_first_two_stages(runner.data_batch['img_dfui'].data[0].cuda())
            self.EF = runner.EF
            loss = self.kl_loss(self.EF, self.DFUI_H1) * self.loss_weight
            runner.outputs = dict(
                loss = loss,
                log_vars = {'kl_loss' : loss.item()},
                num_samples = len(runner.data_batch['img_metas'])
            )
            runner.log_buffer.update(runner.outputs['log_vars'], runner.outputs['num_samples'])