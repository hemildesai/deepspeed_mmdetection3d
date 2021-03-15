# DeepSpeed Integration for [MMDetection3d](https://mmdetection3d.readthedocs.io)

## Setup
- Init and update MMDetection3d submodule
```bash
git submodule init
git submodule update
```
- Install `MMDetection3d` submodule using the instructions at https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation. Remember to install the submodule inside this repository.
- Second, install DeepSpeed like:
```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 ./install.sh
cd ..
```
- Third, install NVIDIA Apex like:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```
- Symlink the Lyft dataset like
```bash
mkdir -p ./data/
# ln -s path/to/lyft/data /path/to/deepspeed_mmdetection3d/data/
ln -s /home/hd10/data/lyft /home/hd10/deepspeed_mmdetection3d/data/
```

## Running
- Run FP16 training like
```bash
nohup deepspeed --num_nodes 1 --num_gpus 1 ds/train.py --work-dir ./outputs_fp16/ --gpus 1 configs/ssn/ssn_lyft_base.py --deepspeed_config configs/ds/ds_config_fp16.json &
```
- Run AMP training like
```bash
nohup deepspeed --num_nodes 1 --num_gpus 1 ds/train.py --work-dir ./outputs_amp/ --gpus 1 configs/ssn/ssn_lyft_base.py --deepspeed_config configs/ds/ds_config_amp.json &
```

For more options look at the DeepSpeed and AMP documentation.