import argparse
import os.path as osp
import pickle
import base64

import mmcv
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_dir = osp.abspath(args.work_dir)
    mmcv.mkdir_or_exist(data_dir)
    dataset = build_dataset(cfg.data.train)
    prog_bar = mmcv.ProgressBar(len(dataset))

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )

    for i, datum in enumerate(dataset):
        if i > 1 and i % 5 == 0:
            ddd = datum["points"].data
            ddd = ddd.view(1, -1, 4)
            vvv, num, coors = model.voxelize(ddd)
            import pdb

            pdb.set_trace()
            break

        # filename = datum["img_metas"].data["pts_filename"].split("/")[-1]
        # filename = osp.splitext(filename)[0]
        # with open(osp.join(data_dir, f"{filename}.pkl"), 'wb') as fp:
        #     pickle.dump(datum, fp, protocol=pickle.HIGHEST_PROTOCOL)

        prog_bar.update()


if __name__ == "__main__":
    main()