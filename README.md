# Fisher-DP-CIFAR

基于Fisher信息矩阵的差分隐私深度学习实现，使用CIFAR-10数据集。

## 项目结构

```
.
├── fisher_dp_cifar.py    # 主程序
├── fisher.py            # Fisher信息矩阵计算
├── model.py             # CNN模型定义
├── utils.py             # 工具函数
└── requirements.txt     # 依赖包
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- numpy 1.19.2+
- scipy 1.7.1+
- tqdm 4.62.3+
- matplotlib 3.4.3+

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/fisher-dp-cifar.git
cd fisher-dp-cifar
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python fisher_dp_cifar.py
```

## 主要功能

1. 使用Fisher信息矩阵计算模型参数的敏感度
2. 实现差分隐私训练
3. 使用影响函数进行模型校准
4. 评估模型在关键样本集上的性能

## 数据集

使用CIFAR-10数据集，程序会自动下载。

## 许可证

MIT License 