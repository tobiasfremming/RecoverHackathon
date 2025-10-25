"""
Monitor model health during training.
"""

import torch
import torch.nn as nn
from typing import Dict


class ModelHealthMonitor:
    """Monitor model parameters and gradients for anomalies."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_norms = {}
        self.grad_norms = {}
        
    def check_parameters(self) -> Dict[str, float]:
        """Check parameter norms."""
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param is not None:
                param_norm = param.data.norm(2).item()
                stats[f"param/{name}"] = param_norm
                
                # Check for extreme values
                if param_norm > 1000:
                    print(f"Warning: Large parameter norm in {name}: {param_norm:.2f}")
                
                # Check for NaN/Inf
                if torch.isnan(param).any():
                    print(f"ERROR: NaN in parameter {name}")
                if torch.isinf(param).any():
                    print(f"ERROR: Inf in parameter {name}")
        
        return stats
    
    def check_gradients(self) -> Dict[str, float]:
        """Check gradient norms."""
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                stats[f"grad/{name}"] = grad_norm
                
                # Check for extreme gradients
                if grad_norm > 10:
                    print(f"Warning: Large gradient in {name}: {grad_norm:.2f}")
                
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    print(f"ERROR: NaN gradient in {name}")
                    return None  # Signal to skip this batch
                if torch.isinf(param.grad).any():
                    print(f"ERROR: Inf gradient in {name}")
                    return None
        
        return stats
    
    def get_summary(self) -> str:
        """Get a summary of model health."""
        total_params = sum(p.numel() for p in self.model.parameters())
        total_param_norm = sum(p.data.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
        
        grad_params = [p for p in self.model.parameters() if p.grad is not None]
        if grad_params:
            total_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in grad_params) ** 0.5
        else:
            total_grad_norm = 0
        
        return f"Params: {total_params:,}, Param norm: {total_param_norm:.4f}, Grad norm: {total_grad_norm:.4f}"
