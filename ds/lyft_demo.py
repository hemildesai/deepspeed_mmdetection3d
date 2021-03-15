from argparse import ArgumentParser

import torch
from deepspeed.profiling.flops_profiler import get_model_profile

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter, MMDataParallel

from mmdet3d.apis import show_result_meshlab
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def init_detector(config, checkpoint=None, device="cuda:0"):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = config.class_names
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def main():
    parser = ArgumentParser()
    # parser.add_argument("pcd", help="Point cloud file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.6, help="bbox score threshold"
    )
    parser.add_argument(
        "--out-dir", type=str, default="demo", help="dir to save results"
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # model = MMDataParallel(model, device_ids=[0])
    device = next(model.parameters()).device
    dataset = build_dataset(cfg.data.test)
    # print(model)

    # test a single image
    # result, data = inference_detector(model, args.pcd)
    data = dataset[10]
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data["img_metas"] = data["img_metas"][0].data
        data["points"] = data["points"][0].data

    # print(data)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    # show the results
    # print(result)
    model.show_results(data, result, args.out_dir)

    with torch.cuda.device(0):
        input_data = data
        input_data["return_loss"] = False
        input_data["rescale"] = True
        macs, params = get_model_profile(
            model=model,  # model
            input_res=(1, 60000, 4),  # input shape or input to the input_constructor
            input_constructor=lambda x: input_data,  # if specified, a constructor taking input_res is used as input to the model
            print_profile=True,  # prints the model graph with the measured profile attached to each module
            detailed=True,  # print the detailed profile
            module_depth=-1,  # depth into the nested modules with -1 being the inner most modules
            top_modules=3,  # the number of top modules to print aggregated profile
            warm_up=10,  # the number of warm-ups before measuring the time of each module
            as_string=True,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            ignore_modules=None, # the list of modules to ignore in the profiling
        )


if __name__ == "__main__":
    main()
