# SNN模型实现

这是一个基于PyTorch和SpikingJelly库实现的脉冲神经网络(SNN)模型，用于MNIST手写数字识别任务。

## 功能特性
- 实现了一个三层的脉冲神经网络模型
- 支持GPU加速训练
- 包含训练、测试和可视化功能
- 动态展示模型处理输入的过程

## 依赖环境
- Python 3.7+
- PyTorch 1.8+
- SpikingJelly
- torchvision
- numpy
- matplotlib

## 安装
```bash
pip install -r requirements.txt
```

## 使用方法
1. 训练模型:
```bash
python snn_model.py
选择选项1进行训练
```

2. 测试模型:
```bash
python snn_model.py
选择选项2进行测试
```

3. 可视化预测过程:
测试过程中会自动展示模型处理输入的可视化结果

## 参数配置
可在`snn_model.py`中修改以下参数:
- `MODEL_HIDDEN_SIZE`: 隐藏层大小
- `TIME_STEPS`: 时间步数
- `BATCH_SIZE`: 批处理大小
- `LEARNING_RATE`: 学习率
- `EPOCHS`: 训练轮数