#!/bin/bash


feature_folders=( 
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/slow_motion/1%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/slow_motion/5%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/slow_motion/10%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt/1%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt/5%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt/10%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt_split/1%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt_split/5%/i3d_features'
                  '/mnt/cephfs/dataset/xiaoyong/thumos14_feature/random_corrupt_split/10%/i3d_features'
)

ckpt=(
    '/mnt/cephfs/dataset/m3lab_data-z/jiaqi/exp/actionformer/cvpr_checkpoint/ckpt_ori/thumos_i3d_random1' # ori model
)

echo "Testing epoch $epoch..."
for ckpt_single in "${ckpt[@]}"
do
  ckptt="$ckpt_single"
  # 遍历上边feature_folders变量的路径，测试模型在不同特征下的性能
  for folder in "${feature_folders[@]}"
  do
    feature_path="$folder"
    python eval.py ./configs/thumos_i3d.yaml \
    ${ckptt} \
    --change_test_path True \
    --test_path ${feature_path}
    sleep 2
  done
  # 下面这里用于计算15个加噪setting测试下的mAP均值，避免手工计算
  file_path="result_i3d.txt"

  # 从文件中提取后15个mAP值，计算它们的平均值
  avg_mAP=$(grep -oP 'mAP\(tIoU=0\.5\) is: \K\d+\.\d+' "$file_path" | tail -n 15 | awk '{sum+=$1} END{print sum/15}')

  # 将平均值附加到文件末尾
  echo "计算的后15个mAP均值为：${avg_mAP}%" >> "$file_path"

  echo "操作完成！"
done

