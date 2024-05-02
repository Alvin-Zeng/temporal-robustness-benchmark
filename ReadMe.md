# Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions

## Introduction

Our paper is accepted to CVPR 2024 and an arXiv version can be found at [this link](https://arxiv.org/abs/2403.20254).

## Changelog

- 02/05/2024: We have uploaded the code for adding corruptions and extracting features on the Thumos14 dataset, including the I3D and VideoMAE V2 extractors. We have also included three example models trained using TRC loss.

## Table of Contents

The code includes:

- the addition of noise to I3D and VideoMAE V2 features on the Thumos dataset, used for extracting features from the Thumos14-C (test set) or training models using TRC loss. 
- Three example models using TRC loss are included: ActionFormer, TriDet, and TemporalMaxer.

## Installation

- Read [INSTALL.md](./INSTALL.md) for installing necessary dependencies.



## Quick Start

### Modify the configuration file

Before running the code, you need to modify the configuration file. For both extracting features and running the TRC loss example code, specific instructions for modifying the configuration file are provided at the beginning of each config file. Parameters that have `$$` in the comments need to be reset, while other parameters can be kept at their default values.

### Run the code

For I3D features on the Thumos dataset, run the command: `python main.py`

For videomaev2 features on the Thumos dataset, run the command: `python extract_tad_feature_thumos.py`

For [ActionFormer](https://github.com/happyharrycn/actionformer_release), [TriDet](https://github.com/dingfengshi/tridet) and [TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer), the command for running the code is consistent with the instructions provided by the original author. Please refer to the GitHub instructions of the original author for details.

## Citation

Please cite the paper in your publications if it helps your research:

```latex
@article{zeng2024benchmarking,
  title={Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions},
  author={Zeng, Runhao and Chen, Xiaoyong and Liang, Jiaming and Wu, Huisi and Cao, Guangzhong and Guo, Yong},
  journal={arXiv preprint arXiv:2403.20254},
  year={2024}
}
```