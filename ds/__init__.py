from deepspeed import __version__, __git_hash__, __git_branch__
from deepspeed.utils import log_dist
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.pipe import PipelineModule

from ds.engine import DSEngine


def initialize(
    args,
    model,
    optimizer=None,
    model_parameters=None,
    training_data=None,
    lr_scheduler=None,
    mpu=None,
    dist_init_required=None,
    collate_fn=None,
    config_params=None,
):
    log_dist(
        "DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
            __version__, __git_hash__, __git_branch__
        ),
        ranks=[0],
    )

    if not isinstance(model, PipelineModule):
        engine = DSEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config_params=config_params,
        )
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=model.mpu(),
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config_params=config_params,
        )

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)