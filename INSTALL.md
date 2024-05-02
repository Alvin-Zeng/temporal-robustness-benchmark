# Extract Thumos14 I3D feature

We extract I3D feature based on [this project](https://github.com/Finspire13/pytorch-i3d-feature-extraction).

## Requirements

- torch >= 1.10.1

- PyYAML 6.0

- simplejson 3.18.4

- numpy 1.23.5

- Pillow 9.4.0

- opencv-python 4.7.0.72

Additionally, to add motion blur corruptions, you will need to use the Wand library, which is recommended to be installed using apt：

```sh
sudo apt-get install libmagickwand-dev
```

# Extract Thumos14 VideoMAE v2 feature

We extract videomae_v2 feature based on  [this project](https://github.com/OpenGVLab/VideoMAEv2/tree/master).

In addition to the environment configured for this project, you will need to install the following Python libraries:

- PyYAML 6.0

- simplejson 

- tqdm

Additionally, to add motion blur corruptions, you will need to use the Wand library, which is recommended to be installed using apt：

```sh
sudo apt-get install libmagickwand-dev
```

# Use TRC loss on ActionFormer, TriDet and TemporalMaxer

We are modifying their code, so you only need to configure the environment according to their original setup without installing additional libraries. Here is the link to their code:

[ActionFormer](https://github.com/happyharrycn/actionformer_release), [TriDet](https://github.com/dingfengshi/tridet), [TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer)

