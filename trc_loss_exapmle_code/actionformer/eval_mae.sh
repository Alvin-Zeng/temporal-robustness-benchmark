#!/bin/bash
feature_folders=( 
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/rgb_features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_1%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_5%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/blackframe_10%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_1%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_5%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/motionblur_10%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_1%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_5%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/occlusion_10%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_1%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_5%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/overexposure_10%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_1%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_5%/features'
                '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/data/thumos/actionformer/videomae_v2/corrupted_rgb_features/test/packetloss_10%/features'
             
 )

ckpt=(
    '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/actionformer/cvpr_checkpoint/ckpt_ori/thumos_mae_random1' 
)

echo "Testing epoch $epoch..."
for ckpt_single in "${ckpt[@]}"
do
  ckptt="$ckpt_single"
  for folder in "${feature_folders[@]}"
  do
    feature_path="$folder"
    python eval.py ./configs/thumos_mae.yaml \
    ${ckptt} \
    --change_test_path True \
    --test_path ${feature_path}
    sleep 2
  done
  ##下面这里用于计算15个加噪setting测试下的mAP均值，避免手工计算
  file_path="result_mae.txt"

  # 从文件中提取后15个mAP值，计算它们的平均值
  avg_mAP=$(grep -oP 'mAP\(tIoU=0\.5\) is: \K\d+\.\d+' "$file_path" | tail -n 15 | awk '{sum+=$1} END{print sum/15}')

  # 将平均值附加到文件末尾
  echo "计算的后15个mAP均值为：${avg_mAP}%" >> "$file_path"

  echo "操作完成！"
done

