import json

# 输入的JSON文件名
input_json_file = 'thumos14.json'
# 输出的JSON文件名
output_json_file = 'thumos14_10fps.json'

# 读取输入的JSON文件
with open(input_json_file, 'r') as file:
    database = json.load(file)

# 处理每个video的segment(frames)数据
for video_id, video_data in database['database'].items():
    video_data['fps'] = 10.0

# 写入到新的JSON文件
with open(output_json_file, 'w') as file:
    json.dump(database, file, indent=4)

print(f"JSON file '{input_json_file}' has been processed and saved as '{output_json_file}'.")
