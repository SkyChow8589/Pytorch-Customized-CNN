import torch

# 加载模型状态字典
model_path = 'D:\\pytorch-cifar-master\\checkpoint\\ckpt.pth'  # 替换为您的文件路径
model_state_dict = torch.load(model_path)

# 打印状态字典的内容
for key, value in model_state_dict.items():
    print(key, value)
