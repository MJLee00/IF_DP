import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import copy
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
from utils import compute_hessian_inverse  # 导入工具函数
from fisher import compute_fisher, compute_user_sensitivity, train_with_dp  # 导入Fisher相关函数
from model import CNN
from influence_function import compute_influence_scores, calibrate_model

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 检查数据集是否存在
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 加载CIFAR-10数据集
try:
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                         download=True, transform=transform)
except Exception as e:
    print(f"下载数据集失败: {e}")
    print("请手动下载CIFAR-10数据集放置在 ./data 目录下")
    print("下载链接: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    exit(1)

# 数据集分割
#indices = list(range(50000))

indices = list(range(500))
np.random.shuffle(indices)
# private_indices = indices[:40000]
# public_indices = indices[40000:]

private_indices = indices[:400]
public_indices = indices[400:500]

private_dataset = Subset(trainset, private_indices)
public_dataset = Subset(trainset, public_indices)


def compute_influence_scores(model, public_loader, train_loader, fisher, device):
    """计算公共池中每个样本的影响分数"""
    model.eval()
    influence_scores = []
    
    # 使用工具函数计算Hessian矩阵的逆和test样本的乘积
    
    for data, target in public_loader:
        # Ensure data and target are on the correct device
        data, target = data.to(device), target.to(device)
        
        # Process each sample in the batch individually
        for i in range(data.size(0)):  # Iterate over batch dimension
            # Extract single sample and target
            single_data = data[i:i+1]  # Keep batch dimension for model compatibility
            single_target = target[i:i+1]
            
            
            gpu_id = device.index if hasattr(device, 'index') else -1
            test_hessian_inv = compute_hessian_inverse(
                z_test=single_data,
                t_test=single_target,
                model=model,
                z_loader=train_loader,  # Still passing full loader for Hessian context
                gpu=gpu_id,
                damp=0.01,
                scale=25.0,
                recursion_depth=1
            )
            
            # Zero gradients before computing loss for this sample
            model.zero_grad()
            output = model(single_data)
            loss = F.cross_entropy(output, single_target)
            loss.backward()
            
            influence_score = 0
            for j, (name, param) in enumerate(model.named_parameters()):
                if param.grad is not None:
                    # Compute influence score: -∇θℓ(s,θDP)⊤H⁻¹∇θℓ(z,θDP)
                    influence_score -= torch.sum(test_hessian_inv[j] * param.grad.data)
            
            influence_scores.append(influence_score.item())
    
    return influence_scores

def calibrate_model(model, public_loader, train_loader, fisher, top_k=100, device='cuda:3'):
    """使用影响函数校准模型"""
    model.eval()
    
   
    # 计算影响分数
    print("计算影响分数...")
    influence_scores = compute_influence_scores(model, public_loader, train_loader, fisher, device)
    
    # 选择top-k个最有帮助的样本
    top_indices = np.argsort(influence_scores)[:top_k]
    print(f"选择top-{top_k}个最有帮助的样本")
    
    # 计算确定性偏差
    print("计算确定性偏差...")
    calibration = {}
    for name, param in model.named_parameters():
        calibration[name] = torch.zeros_like(param.data)
    
    # 计算加权梯度和的H⁻¹
    for idx in top_indices:
        data, target = next(iter(public_loader))
        data, target = data[idx:idx+1].to(device), target[idx:idx+1].to(device)

        gpu_id = device.index if hasattr(device, 'index') else -1
        test_hessian_inv = compute_hessian_inverse(
            z_test=data,
            t_test=target,
            model=model,
            z_loader=train_loader,  # Still passing full loader for Hessian context
            gpu=gpu_id,
            damp=0.01,
            scale=25.0,
            recursion_depth=1
        )
        
        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                # 计算 -1/n * H⁻¹∇θℓ(z,θDP)
                calibration[name] -= param.grad.data * test_hessian_inv[i] / top_k
    
    # 应用校准：θ*DP = θDP + Δθw
    print("应用校准...")
    for name, param in model.named_parameters():
        param.data += calibration[name]
    
    return model

def compute_utility_drop(baseline_model, dp_model, calibrated_model, critical_loader, device):
    """计算效用下降
    
    计算关键样本集上的损失差异：
    ΔLDP = 1/|Scrit| * ∑_{s∈Scrit} [ℓ(s,θDP) - ℓ(s,θclean)]
    """
    models = {
        "基线模型": baseline_model,
        "DP模型": dp_model,
        "校准DP模型": calibrated_model
    }
    
    results = {}
    for name, model in models.items():
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in critical_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                total_loss += loss.item() * len(data)
                total_samples += len(data)
        
        results[name] = total_loss / total_samples
    
    # 计算效用下降
    dp_drop = results["DP模型"] - results["基线模型"]
    calibrated_drop = results["校准DP模型"] - results["基线模型"]
    
    print("\n效用下降分析:")
    print(f"DP模型效用下降: {dp_drop:.4f}")
    print(f"校准DP模型效用下降: {calibrated_drop:.4f}")
    print(f"改善程度: {(dp_drop - calibrated_drop) / dp_drop * 100:.2f}%")
    
    return results

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 创建数据加载器
    private_loader = DataLoader(private_dataset, batch_size=128, shuffle=True)
    public_loader = DataLoader(public_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    
    # 创建关键样本集（所有猫的图片）
    cat_indices = [i for i, (_, label) in enumerate(testset) if label == 3]  # 3是猫的类别
    critical_dataset = Subset(testset, cat_indices)
    critical_loader = DataLoader(critical_dataset, batch_size=128, shuffle=False)
    
    # 训练基线模型
    print("训练基线模型...")
    baseline_model = CNN().to(device)
    optimizer = torch.optim.SGD(baseline_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        baseline_model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(private_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(private_loader)
        scheduler.step(avg_loss)
    
    # 计算Fisher信息矩阵
    print("计算Fisher信息矩阵...")  
    fisher = compute_fisher(baseline_model, private_loader, device)
    
    # 训练DP模型
    print("训练差分隐私模型...")
    dp_model = CNN().to(device)
    dp_model = train_with_dp(dp_model, private_loader, fisher, epsilon=1.0, delta=1e-5, device=device)
    
    # 校准DP模型
    print("校准差分隐私模型...")
    calibrated_dp_model = calibrate_model(dp_model, public_loader, private_loader, fisher, top_k=100, device=device)
    
    # 评估所有模型
    print("评估模型性能...")
    models = {
        "基线模型": baseline_model,
        "DP模型": dp_model,
        "校准DP模型": calibrated_dp_model
    }
    
    results = {}
    for name, model in models.items():
        model.eval()
        correct = 0
        total = 0
        cat_correct = 0
        cat_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 计算猫类别的准确率
                cat_mask = (target == 3)
                cat_correct += (predicted[cat_mask] == target[cat_mask]).sum().item()
                cat_total += cat_mask.sum().item()
        
        results[name] = {
            "总体准确率": 100. * correct / total,
            "猫类别准确率": 100. * cat_correct / cat_total
        }
    
    # 打印结果
    for name, metrics in results.items():
        print(f"\n{name}性能:")
        print(f"总体准确率: {metrics['总体准确率']:.2f}%")
        print(f"猫类别准确率: {metrics['猫类别准确率']:.2f}%")

    # 计算效用下降
    utility_drop_results = compute_utility_drop(baseline_model, dp_model, calibrated_dp_model, critical_loader, device)

if __name__ == "__main__":
    main() 