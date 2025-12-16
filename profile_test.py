import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 场景设置：现代 CNN + 模拟的数据瓶颈
# ==========================================


class SyntheticImageDataset(Dataset):
    def __init__(self, size=5000):
        self.size = size
        # 预先生成一些随机数据
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 1000, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # [模拟瓶颈]：
        # 在这里我们不加 time.sleep，而是通过设置 DataLoader 的 num_workers=0
        # 且 batch_size 较小，来模拟 CPU 处理/调度跟不上 GPU 的情况。
        # 现代 GPU (如 3090/4090/A100) 跑 ResNet18 非常快，极易出现这种瓶颈。
        return self.data[idx], self.labels[idx]


def modern_profiling_experiment():
    device = torch.device("cuda")

    # 使用标准的 ResNet18
    model = models.resnet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # batch_size=32 对于 ResNet18 来说很小，会让 GPU 算得飞快，然后等待下一批数据
    dataset = SyntheticImageDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    print("🚀 开始现代化 Profiling (PyTorch Kineto)...")

    # ==========================================
    # 2. 现代 Profiler 配置
    # ==========================================
    # schedule: 自动管理 warmup, active 周期，避免分析器本身的开销影响结果
    my_schedule = schedule(skip_first=1, wait=1, warmup=1, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/modern_test"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # 捕捉 Python 代码堆栈，方便定位是哪行代码卡住了
    ) as p:
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= 6:
                break  # 只需要跑几个 step 即可

            with record_function("Data_Transfer_H2D"):  # 手动打标签，方便在图中识别
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with record_function("Model_Forward"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with record_function("Model_Backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            p.step()  # 通知 profiler 进入下一阶段

    print("✅ Profiling 完成。数据已保存至 ./log/modern_test")
    print("请按照下文指引使用 Perfetto 进行可视化分析。")


if __name__ == "__main__":
    modern_profiling_experiment()
