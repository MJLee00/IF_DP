import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import compute_hessian_inverse

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
