#!/bin/bash
##这个脚本是用来测试不同的特征提取方法的，需要修改的地方有：测试使用的特征路径(feature_folders)、测试使用的模型路径(ckpt)、测试使用的配置文件路径(configs/thumos_i3d.yaml)、并且指定
feature_folders=( 
                  # '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/rgb_features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/blackframe_1%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/blackframe_5%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/blackframe_10%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/motionblur_1%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/motionblur_5%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/motionblur_10%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/occlusion_1%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/occlusion_5%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/occlusion_10%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/overexposure_1%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/overexposure_5%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/overexposure_10%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/packetloss_1%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/packetloss_5%/features'
                  '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/i3d/corrupted_rgb_features/test/packetloss_10%/features'
 )

echo "Testing epoch $epoch..."
# for ckpt_single in "${ckpt[@]}"
# do
  # ckptt="$ckpt_single"
for folder in "${feature_folders[@]}"
do
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py \
configs/gtad/thumos_i3d.py \
--checkpoint pretrained/gtad_thumos_i3d_sw256_epoch_6_fc380b36.pth \

  feature_path="$folder"
  CUDA_VISIBLE_DEVICES=5,6 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py \
  configs/gtad/thumos_i3d.py \
  --checkpoint pretrained/gtad_thumos_i3d_sw256_epoch_6_fc380b36.pth \
  --test_path ${feature_path}
  sleep 2
done
#   ##下面这里用于计算15个加噪setting测试下的mAP均值，避免手工计算
#   file_path="result_i3d.txt"

#   # 从文件中提取后15个mAP值，计算它们的平均值
#   avg_mAP=$(grep -oP 'mAP\(tIoU=0\.5\) is: \K\d+\.\d+' "$file_path" | tail -n 15 | awk '{sum+=$1} END{print sum/15}')

#   # 将平均值附加到文件末尾
#   echo "计算的15个setting下noise的mAP均值为：${avg_mAP}%" >> "$file_path"

#   echo "操作完成！"
# done

