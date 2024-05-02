import pickle

# 指定.pkl文件的路径
file_path = "/mnt/cephfs/home/zengrunhao/chenxiaoyong/TemporalMaxer/ckpt/i3d/thumos/eval_results.pkl"

# 使用pickle模块加载.pkl文件
with open(file_path, "rb") as file:
    loaded_object = pickle.load(file)

# 现在，loaded_object中包含了.pkl文件中保存的对象
# 您可以使用loaded_object进行进一步的操作

# 示例：打印加载的对象
print(loaded_object)
print(1)