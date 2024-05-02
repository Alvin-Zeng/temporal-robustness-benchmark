import re

log_content = """
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_1%/features
mAP(tIoU=0.5) is: 67.57%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_5%/features
mAP(tIoU=0.5) is: 48.18%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_10%/features
mAP(tIoU=0.5) is: 28.73%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_1%/features
mAP(tIoU=0.5) is: 73.56%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_5%/features
mAP(tIoU=0.5) is: 72.95%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_10%/features
mAP(tIoU=0.5) is: 70.81%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_1%/features
mAP(tIoU=0.5) is: 71.84%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_5%/features
mAP(tIoU=0.5) is: 67.66%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_10%/features
mAP(tIoU=0.5) is: 62.78%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_1%/features
mAP(tIoU=0.5) is: 71.09%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_5%/features
mAP(tIoU=0.5) is: 66.34%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_10%/features
mAP(tIoU=0.5) is: 61.72%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_1%/features
mAP(tIoU=0.5) is: 73.42%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_5%/features
mAP(tIoU=0.5) is: 73.36%
ckpt is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/tridet/cvpr_checkpoint/ckpt_jsd/thumos_videomae_random1_packetloss_w_1/bestmodel.pth.tar
dataset is: /mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_10%/features
mAP(tIoU=0.5) is: 72.09%
"""

# 利用正则表达式匹配每组mAP
pattern = re.compile(r'mAP\(tIoU=0\.5\) is: (\d+\.\d+)%', re.MULTILINE)
matches = pattern.findall(log_content)

# 每组三个mAP的均值
averages = [sum(map(float, matches[i:i+3])) / 3.0 for i in range(0, len(matches), 3)]

# 输出每组均值
for avg in averages:
    print(f'Mean mAP for a group of three: {avg:.2f}%')
