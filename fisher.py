import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
from utils import compute_hessian_inverse
def compute_fisher(model, dataloader, device):
    """计算Fisher信息矩阵"""
    model.eval()
    fisher = {}
    
    # 初始化Fisher矩阵
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)
    
    # 计算梯度并累加
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2
    
    # 计算平均值
    for name in fisher:
        fisher[name] /= len(dataloader)
    
    return fisher

def compute_top_k_eigenvectors(fisher, k=256):
    """计算Fisher矩阵的top-k特征向量"""
    # 将Fisher矩阵展平并连接
    flat_fisher = []
    for name, param in fisher.items():
        flat_fisher.append(param.view(-1))
    flat_fisher = torch.cat(flat_fisher)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eigh(flat_fisher.cpu().numpy(), subset_by_index=[0, k-1])
    return eigenvalues, eigenvectors

def compute_user_sensitivity(model, train_loader, fisher_inv, device):
    """计算用户级敏感度Δ2
    
    计算两个相邻批次（只相差一个用户的数据）在Mahalanobis范数下的最大差异
    Δ2 = max_{B≃B'} ||g(B) - g(B')||_{F^-1}
    """
    model.eval()
    max_sensitivity = 0.0
    
    # 获取所有用户的数据
    all_data = []
    all_targets = []
    for data, target in train_loader:
        all_data.append(data)
        all_targets.append(target)
    
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算完整批次的梯度
    model.zero_grad()
    output = model(all_data.to(device))
    loss = F.cross_entropy(output, all_targets.to(device))
    loss.backward()
    
    full_grad = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            full_grad[name] = param.grad.data.clone()
    
    # 对每个用户，计算移除该用户后的梯度差异
    for i in range(len(all_data)):
        # 创建不包含第i个用户的批次
        mask = torch.ones(len(all_data), dtype=torch.bool)
        mask[i] = False
        batch_data = all_data[mask]
        batch_targets = all_targets[mask]
        
        # 计算新批次的梯度
        model.zero_grad()
        output = model(batch_data.to(device))
        loss = F.cross_entropy(output, batch_targets.to(device))
        loss.backward()
        
        # 计算梯度差异的Mahalanobis范数
        diff_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_diff = full_grad[name] - param.grad.data
                # 计算Mahalanobis范数 ||g(B) - g(B')||_{F^-1}
                diff_norm += torch.sum(grad_diff * fisher_inv[name] * grad_diff)
        
        diff_norm = torch.sqrt(diff_norm)
        max_sensitivity = max(max_sensitivity, diff_norm.item())
    
    return max_sensitivity

def train_with_dp(model, train_loader, fisher, epsilon, delta, device):
    """使用差分隐私训练模型"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
    # 计算Fisher矩阵的逆
    print("计算Fisher矩阵的逆...")
    fisher_inv = {}
    for name, param in fisher.items():
        # 添加一个小的值以避免数值不稳定性
        fisher_matrix = param + 1e-6 
        try:
            fisher_inv[name] = torch.linalg.inv(fisher_matrix)
        except RuntimeError:
            # 如果矩阵不是方阵，使用伪逆
            if fisher_matrix.dim() == 1:
                fisher_inv[name] = torch.linalg.pinv(fisher_matrix.view(1,-1))
            else:
                fisher_inv[name] = torch.linalg.pinv(fisher_matrix)
    # 使得fisher_inv和param的维度相同
    for name, param in fisher.items():
        fisher_inv[name] = fisher_inv[name].view_as(param)
        
    # 计算用户级敏感度Δ2
    print("计算用户级敏感度...")
    delta2 = compute_user_sensitivity(model, train_loader, fisher_inv, device)
    print(f"用户级敏感度Δ2: {delta2:.4f}")
    
    # 计算噪声尺度
    sigma = np.sqrt(2 * np.log(1.25/delta)) / epsilon
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 计算梯度 g(B) = 1/|B| * ∑∇θℓ(z,θ)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # 应用Mahalanobis裁剪
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 计算Mahalanobis范数 ||g||_F^-1 = √(g^T F^-1 g)
                grad = param.grad.data
                mahalanobis_norm = torch.sqrt(torch.sum(grad * fisher_inv[name] * grad))
                
                # 如果范数大于敏感度Δ2，进行缩放
                if mahalanobis_norm > delta2:
                    param.grad.data = param.grad.data * (delta2 / mahalanobis_norm)
                
                # 添加各向异性噪声: N(0, σ²Δ²F^-1)
                noise = torch.randn_like(param.grad.data)
                # 使用Fisher矩阵的逆来缩放噪声
                scaled_noise = noise * fisher_inv[name] * sigma * delta2
                param.grad.data += scaled_noise
        
        optimizer.step()
    


    return model