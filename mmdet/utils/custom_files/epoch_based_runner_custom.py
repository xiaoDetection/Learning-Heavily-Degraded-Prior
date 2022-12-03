import time

from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

from torchvision import transforms

@RUNNERS.register_module()
class EpochBasedRunnerCustom(EpochBasedRunner):
    def __init__(self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, meta=None, max_iters=None, max_epochs=None):
        super().__init__(model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)
        self.train_RFTM = False

    def train(self, data_loader, **kwargs):
            self.model.train()
            self.mode = 'train'
            self.data_loader = data_loader
            self._max_iters = self._max_epochs * len(self.data_loader)
            self.call_hook('before_train_epoch')
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            topil = transforms.ToPILImage()
            for i, data_batch in enumerate(self.data_loader):
                self.data_batch = data_batch
                self._inner_iter = i
                self.call_hook('before_train_iter')
                if self.train_RFTM:
                    # HD_u forward
                    self.EF = self.model.module.backbone.forward_first_two_stages(data_batch['img_masked'].data[0].cuda(), True)
                else:
                    self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1

            self.call_hook('after_train_epoch')
            self._epoch += 1