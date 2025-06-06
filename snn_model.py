import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, layer, functional

# 全局模型参数
MODEL_HIDDEN_SIZE = 4096
TIME_STEPS = 30
BATCH_SIZE = 2048  # 增加batch_size以充分利用GPU内存
LEARNING_RATE = 0.0005
EPOCHS = 20
INPUT_SIZE = 28*28  # MNIST图像大小
OUTPUT_SIZE = 10
global device

class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=TIME_STEPS):
        super(SNNModel, self).__init__()
        self.time_steps = time_steps
        
        # 输入层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 脉冲神经元层
        self.lif1 = neuron.LIFNode(tau=2.0)
        
        # 隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = neuron.LIFNode(tau=2.0)
        
        # 输出层
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.lif3 = neuron.LIFNode(tau=2.0)
        
    def forward(self, x):
        # 初始化膜电位
        functional.reset_net(self)
        
        # 确保输入形状正确
        x = x.view(x.size(0), -1)
        
        # 将图像分成多个块，模拟逐步输入
        chunk_size = x.size(1) // self.time_steps
        
        # 时间步循环
        outputs = []
        accumulated_input = torch.zeros_like(x)
        
        for t in range(self.time_steps):
            # 获取当前时间步的图像块
            start_idx = t * chunk_size
            end_idx = (t + 1) * chunk_size if t < self.time_steps - 1 else x.size(1)
            current_chunk = x[:, start_idx:end_idx]
            
            # 累积输入信息
            accumulated_input = accumulated_input.clone()
            accumulated_input[:, start_idx:end_idx] = current_chunk
            
            # 前向传播
            x_t = self.fc1(accumulated_input)
            x_t = self.lif1(x_t)
            
            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)
            
            x_t = self.fc3(x_t)
            x_t = self.lif3(x_t)
            
            outputs.append(x_t)
        
        # 返回最后一个时间步的输出
        return outputs[-1]

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    model.to(device)
    
    # 计时相关变量
    start_time = None
    end_time = None
    
    # 检查CUDA可用性并初始化计时器
    if str(device) == "cuda":
        # 创建CUDA事件用于计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 预热GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 打印GPU信息
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        print(f"CUDA utilization: {torch.cuda.utilization(0)}%")
    else:
        # CPU计时使用time模块
        import time
        start_time = time.time()
    
    total_time = 0.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        if str(device) == "cuda":
            start_event.record()
        else:
            start_time = time.time()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        if str(device) == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed_time = time.time() - start_time
        total_time += elapsed_time
        print(f'Epoch {epoch+1} completed in {elapsed_time:.2f}s, Total time: {total_time:.2f}s')

def main():
    device = select_device()
    # 超参数
    input_size = INPUT_SIZE
    hidden_size = MODEL_HIDDEN_SIZE
    output_size = OUTPUT_SIZE
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    epochs = EPOCHS
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
    
    # 初始化模型
    model = SNNModel(input_size, hidden_size, output_size).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, epochs)
    
    # 保存模型
    torch.save(model.state_dict(), 'snn_model.pth')
    print("Model saved to snn_model.pth")

def test_model(model, test_loader, device=None, num_batches=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    # 设备监控初始化
    if str(device) == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        print(f"CUDA utilization: {torch.cuda.utilization(0)}%")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
                
            # 使用non_blocking传输减少等待时间
            images = images.view(images.size(0), -1).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每100个batch打印一次进度和设备状态
            if batch_idx % 100 == 0:
                print(f'Processed {batch_idx+1} batches, current accuracy: {100 * correct / total:.2f}%')
                if str(device) == "cuda":
                    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                    print(f"CUDA utilization: {torch.cuda.utilization(0)}%")
    
    accuracy = 100 * correct / total
    print(f'\nFinal Test Accuracy: {accuracy:.2f}% (Tested on {total} samples)')
    return accuracy

def visualize_predictions(model, test_loader, device=None, num_images=5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        # 获取所有时间步的输出
        functional.reset_net(model)
        x = images.view(images.size(0), -1).to(device)
        outputs = []
        accumulated_input = torch.zeros_like(x)
        chunk_size = x.size(1) // model.time_steps
        
        # 创建动画
        plt.figure(figsize=(15, 6))
        plt.ion()  # 开启交互模式
        
        for t in range(model.time_steps):
            start_idx = t * chunk_size
            end_idx = (t + 1) * chunk_size if t < model.time_steps - 1 else x.size(1)
            current_chunk = x[:, start_idx:end_idx]
            
            accumulated_input = accumulated_input.clone()
            accumulated_input[:, start_idx:end_idx] = current_chunk
            
            x_t = model.fc1(accumulated_input)
            x_t = model.lif1(x_t)
            x_t = model.fc2(x_t)
            x_t = model.lif2(x_t)
            x_t = model.fc3(x_t)
            x_t = model.lif3(x_t)
            
            outputs.append(x_t)
            
            # 动态显示处理过程
            plt.clf()
            
            # 显示当前输入图像块
            for i in range(num_images):
                plt.subplot(3, num_images, i+1)
                
                # 显示当前输入的部分图像
                partial_img = torch.zeros_like(images[i])
                partial_img.view(-1)[:end_idx] = images[i].view(-1)[:end_idx]
                plt.imshow(partial_img.numpy().squeeze(), cmap='gray')
                plt.title(f'Input Step {t+1}/{model.time_steps}')
                plt.axis('off')
            
            # 显示中间层激活
            for i in range(num_images):
                plt.subplot(3, num_images, num_images+i+1)
                plt.bar(range(10), outputs[t][i].detach().cpu().numpy(), color='blue')
                plt.title(f'Step {t+1} Output')
                plt.ylim([-1, 1])
            
            # 显示当前预测
            _, current_pred = torch.max(outputs[t], 1)
            for i in range(num_images):
                plt.subplot(3, num_images, 2*num_images+i+1)
                plt.text(0.5, 0.5, f'Pred: {current_pred[i].item()}\nTrue: {labels[i].item()}', 
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
            
            plt.pause(0.5)  # 暂停0.5秒
        
        plt.ioff()  # 关闭交互模式
        plt.show()
        
        # 获取最终预测
        _, predicted = torch.max(outputs[-1], 1)
        
        # 显示最终结果
        plt.figure(figsize=(15, 4))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(images[i].numpy().squeeze(), cmap='gray')
            plt.title(f'Final Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def select_device():
    while True:
        print("\n请选择运行设备:")
        print("1. 自动选择(优先使用GPU)")
        print("2. 强制使用CPU")
        print("3. 强制使用GPU(如果可用)")
        device_choice = input("请输入选项(1/2/3): ")
        
        if device_choice == "1":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            break
        elif device_choice == "2":
            device = torch.device("cpu")
            break
        elif device_choice == "3":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                break
            else:
                print("错误: GPU不可用，请重新选择！")
        else:
            print("无效输入，请重新选择！")
    
    print(f"\n使用设备: {device}")
    if str(device) == "cuda":
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    return device

if __name__ == "__main__":
    while True:
        print("\n请选择操作模式:")
        print("1. 训练模型")
        print("2. 验证模型")
        print("3. 退出")
        choice = input("请输入选项(1/2/3): ")
        
        if choice == "1":
            # 训练模式
            main()
            print("训练完成！")
        elif choice == "2":
            # 验证模式
            device = select_device()
            if not os.path.exists('snn_model.pth'):
                print("错误: 未找到训练好的模型文件'snn_model.pth'，请先训练模型！")
                continue
                
            # 加载测试数据集
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            # 设置随机种子以保证可重复性
            torch.manual_seed(42)
            np.random.seed(42)
            
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE//16, shuffle=True, pin_memory=True, num_workers=4)
            
            # 加载训练好的模型
            model = SNNModel(28*28, MODEL_HIDDEN_SIZE, 10).to(device)
            model.load_state_dict(torch.load('snn_model.pth'))
            
            # 测试模型
            test_model(model, test_loader, device)
            
            # 可视化预测结果
            visualize_predictions(model, test_loader, device)
        elif choice == "3":
            print("程序退出。")
            break
        else:
            print("无效输入，请重新选择！")