from collections.abc import Callable, Iterable
from jaxtyping import Float, Int
from typing import Optional
import torch
import math
import os

 
class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW optimizer.
    Args:
        lr : learning rate
        betas: moments params
        wright_decay: weight decay param
    """
    def __init__(self, 
                 params, 
                 lr: float = 1e-3, 
                 betas: tuple[float,float] = (0.9,0.995), 
                 weight_decay: float = 0.1, 
                 eps:float = 1e-8
                ):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}") 
        if betas[0] < 0 or betas[0] >= 1:
            raise ValueError(f"Invalid beta_1: {betas[0]}")  
        if betas[1] < 0 or betas[1] >= 1:
            raise ValueError(f"Invalid beta_2: {betas[1]}") 
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}") 
        defaults = {"lr": lr, "beta_1" : betas[0], "beta_2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()        
        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]   
            beta_1, beta_2 = group["beta_1"], group["beta_2"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue                    
                state = self.state[p] # Get state associated with p.
                
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)                    
                t = state.get("t", 0) + 1                    
                grad = p.grad.data
                
                state["m"] = state["m"]* beta_1 + (1 - beta_1) * grad
                state["v"] = state["v"] * beta_2 + (1 - beta_2) * (grad * grad)
                lr_t = lr * math.sqrt(1 - beta_2 ** t)/(1 - beta_1 ** t)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps) 
                p.data = p.data - p.data * weight_decay * lr 
                state["t"] = t
        return loss


