import argparse
import os.path as osp
import pickle

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process dataset")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")

    args = parser.parse_args()

    return args


# This will take up a lot of data. Only run this if you have free space of >200gb
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_dir = osp.abspath(args.work_dir)
    mmcv.mkdir_or_exist(data_dir)
    dataset = build_dataset(cfg.data.train)
    prog_bar = mmcv.ProgressBar(len(dataset))

    for _, datum in enumerate(dataset):
        filename = datum["img_metas"].data["pts_filename"].split("/")[-1]
        filename = osp.splitext(filename)[0]
        with open(osp.join(data_dir, f"{filename}.pkl"), "wb") as fp:
            pickle.dump(datum, fp, protocol=pickle.HIGHEST_PROTOCOL)

        prog_bar.update()


if __name__ == "__main__":
    main()