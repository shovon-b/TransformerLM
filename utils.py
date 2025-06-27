import torch
import math
import numpy.typing as npt
import random
from collections.abc import Iterable
from einops import einsum, rearrange
import os

def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    """
    Returns the output of softmaxing the given `dim` of the input.
    """
    x_max, _ = torch.max(x, dim, keepdim=True)
    exp_x = torch.exp(x - x_max)
    norm = torch.sum(exp_x, dim, keepdim=True)
    return exp_x / norm
    

def silu(x:torch.Tensor) -> torch.Tensor:
    """Returns the output of applying SiLU(x) = x * sigmoid(x) 
    """
    return x * torch.sigmoid(x)


def cross_entropy_loss(x:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    """
    Given x: Tensor[batch seq_len d_mode] and targets: Tensor[batch seq_len],
    returns the cross-entropy loss: Tensor[float]
    """    
    b, seq_len, _ = x.shape
    x = rearrange(x, "b seq_len ... -> (b seq_len) ..." , b=b, seq_len=seq_len)
    targets =  rearrange(targets, "b seq_len->(b seq_len)", b=b, seq_len=seq_len)
    
    x_max, _ = torch.max(x, -1, keepdim=True)
    log_softmax = (x-x_max) - torch.log(torch.sum(torch.exp(x-x_max), -1, keepdim=True))
    return -log_softmax[torch.arange(b * seq_len), targets].mean()


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, samples language modeling input sequences and their corresponding
    labels from the dataset.
    Args:
        dataset_name : name of the dataset in ".npy" format. Assumed to be located in /data/ dir
        batch_size: batch_size to be fetched
        context_length: sequence length to be fetched
        device: torch device
    """   
    max_start_idx = len(dataset) - context_length         
    start = random.choices(range(max_start_idx), k=batch_size)
    sample_list = []
    pred_list = []
    for i in start:        
        sample = dataset[i : i + context_length]        
        pred = dataset[i + 1 : i + context_length + 1]
        sample_list.append(torch.tensor(sample, dtype=torch.long, device=device))
        pred_list.append(torch.tensor(pred, dtype=torch.long, device=device))    
    x = torch.stack(sample_list) 
    y = torch.stack(pred_list)
    return x, y

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters are modified in-place.
    """
    if not isinstance(parameters, (list, tuple)):
        parameters = list(parameters) 

    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)

    
    with torch.no_grad():
        if total_norm > max_l2_norm:            
            clip_coeff = max_l2_norm / (total_norm + 1e-6) 
            for g in grads:
                g.mul_(clip_coeff)



def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Args:
        it: Iteration number to get learning rate for.
        max_learning_rate: the maximum learning rate for cosine learning rate schedule.
        min_learning_rate: alpha_min, the minimum learning rate.
        warmup_iters: the number of iterations for linear warm-up.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration.
    """
    
    if it < warmup_iters: lr = it * max_learning_rate / warmup_iters
    elif it < cosine_cycle_iters: lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else: lr = min_learning_rate
    return lr


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_filename: str,
):
    """
    Given a model, optimizer, and an iteration number, saves them to models/"out_filename_iteration.pt"

    Args:
        model : the state of this model.
        optimizer: Serialize the state of this optimizer.
        iteration: Serialize this value, which represents the number of training terations.
        out: filename serialize the model, optimizer, and iteration to.
    """
    checkpoints = {"iteration": iteration, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dir_name = "models"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)    
    out_path = os.path.join(dir_name, out_filename + "_"+ f"{iteration}" + ".pt")
    torch.save(checkpoints, out_path)   
    

    
def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
)-> int:
    """
    Given a saved model name with .pt extension "src.pt", restores the saved state to the given model, optimizer and eturn the number of iterations that we previously saved in the checkpoint. The model is expected to be located in models/

    Args:
        src(str of filepath): saved filename.
        model: Restore the state of this model.
        optimizer: Restore the state of this optimizer.
    Returns:
        int: the previously-saved number of iterations.
    """
    if os.path.exists(src):
        src_path = src
    else:
        dir_name= "models"
        src_path = os.path.join(dir_name, src)
    saved_checkpoints = torch.load(src_path)
    model.load_state_dict(saved_checkpoints["model"])
    optimizer.load_state_dict(saved_checkpoints["optimizer"])
    print("Model loaded.")
    return saved_checkpoints["iteration"]

