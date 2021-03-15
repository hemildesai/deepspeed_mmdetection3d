# DeepSpeed Integration for [MMDetection3d](https://mmdetection3d.readthedocs.io)

## Setup
- First install `MMDetection3d` using the instructions at https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation
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