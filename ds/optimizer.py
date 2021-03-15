from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DSOptimizerHook(Hook):
    def after_train_iter(self, runner):
        runner.model.backward(runner.outputs["loss"])
        runner.model.step()