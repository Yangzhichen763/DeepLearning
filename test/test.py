
import torch


model_path = r'./models/best.pt'

state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 检查加载结果类型
if isinstance(state_dict, torch.nn.Module):
    # 如果是 PyTorch 模型，则进行下一步操作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = state_dict.to(device).float()  # 移动模型到设备并转换为浮点数类型
else:
    # 如果加载结果不是 PyTorch 模型，输出错误信息或者处理异常情况
    print(f"Error: Loaded model is not an instance of torch.nn.Module. {type(state_dict)}")


