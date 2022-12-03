from mmcv.runner.optimizer import DefaultOptimizerConstructor

from mmcv.utils import build_from_cfg
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS

def filter1(output):
    if 'layer3' in output[0] or 'layer4' in output[0] or 'backbone' not in output[0]:
        return True
    return False

def filter2(output):
    if 'RFTM' in output[0]:
        return True
    return False

@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructorCustom(DefaultOptimizerConstructor):
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg, paramwise_cfg)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg_finetune = self.optimizer_cfg.copy()['optimizer_finetune']
        optimizer_cfg_RFTM = self.optimizer_cfg.copy()['optimizer_RFTM']
        # if no paramwise option is specified, just use the global setting
        assert self.paramwise_cfg is not None, 'This parameter is not supported.'
        optimizer_cfg_finetune['params'] = map(lambda out : out[1], filter(filter1, model.named_parameters()))
        optimizer_cfg_RFTM['params'] = map(lambda out : out[1], filter(filter2, model.named_parameters()))
        return dict(
            optimizer_finetune = build_from_cfg(optimizer_cfg_finetune, OPTIMIZERS),
            optimizer_RFTM = build_from_cfg(optimizer_cfg_RFTM, OPTIMIZERS)
        )