#!/bin/bash

# 指定txt文件路径
file_path="result.txt"

# 从文件中提取后15个mAP值，计算它们的平均值
avg_mAP=$(grep -oP 'mAP\(tIoU=0\.5\) is: \K\d+\.\d+' "$file_path" | tail -n 15 | awk '{sum+=$1} END{print sum/15}')

# 将平均值附加到文件末尾
echo "计算的后15个mAP均值为：${avg_mAP}%" >> "$file_path"

echo "操作完成！"


