# PowerSAM: Edge-Efficient Segment Anything for Power Systems Through Visual Model Distillation

[ [Paper](https://github.com/fudan-birlab/PowerSAM) ] [ [Project Page](https://github.com/fudan-birlab/PowerSAM) ]

Authors: Nannan Yan, Yuhao Li, Yingke Mao, Xiao Yu, Wenhao Guan, Jiawei Hou and
Taiping Zeng

![Realtime SAM](figs/realtime_sam.png)

**Tl;dr** PowerSAM is proposed as a real-time semantic segmentation framework for edge devices, addressing the challenges of power system equipment inspection, including labor intensity, costs, and human error. By leveraging knowledge distillation from large models to compact backbones and integrating a bounding box prompt generator with a segmentation model, PowerSAM significantly reduces computational complexity while maintaining high segmentation accuracy.

## Installation

To set up the environment for PowerSAM, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/fudan-birlab/PowerSAM.git
```

2. Create and activate a new conda environment:
```sh
conda create -n powersam python=3.8 -y
conda activate powersam
```

3. Install PyTorch and related packages:
```sh
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# or
conda install -y pytorch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install mmdetection dependencies:
```sh
pip install -U openmim
mim install mmengine==0.10.3
mim install mmcv==2.0.0rc4
mim install mmdet==3.3.0
```

5. Install the required Python packages:
```sh
pip install -r requirements.txt
```

6. Install PowerSAM:
```sh
pip install -e .
```

## Getting Started

To get started with PowerSAM, you can follow the example provided in the [Getting Started with PowerSAM](notebooks/powersam_demo.ipynb). This notebook demonstrates how to use the PowerSAM model for segmentation in power system scenarios.


## Acknowledgement

We would like to acknowledge the following projects and their contributions to the development of our work:

- **[SAM](https://github.com/facebookresearch/segment-anything)** with [Apache License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
- **[SAM2](https://github.com/facebookresearch/segment-anything)** with [Apache License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)
- **[SlimSAM](https://github.com/czg1225/SlimSAM)** with [Apache License](https://github.com/czg1225/SlimSAM/blob/master/LICENSE)
- **[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)** with [S-Lab License 1.0](https://github.com/chongzhou96/EdgeSAM/blob/master/LICENSE)
