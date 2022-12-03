from mmcv.runner.hooks.optimizer import OptimizerHook
from mmcv.runner.hooks import HOOKS

@HOOKS.register_module()
class OptimizerHookCustom(OptimizerHook):
    def __init__(self, grad_clip=None):
        super().__init__(grad_clip)

    def before_train_epoch(self, runner):
        # select optimizer
        self.optimizer = runner.optimizer['optimizer_finetune'] if not runner.train_RFTM else runner.optimizer['optimizer_RFTM']

    def after_train_iter(self, runner):
        self.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        self.optimizer.step()