from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn
import math
from einops import einsum, rearrange
from utils import *


class Linear(nn.Module):   
    """
    Performs a linear transformation.
    Uses Xavier initialization for the weight truncated at (-3*std, 3*std)
    Does not include a bias term following the modern LLMs.
    args:
        in_features (int): input dimension.
        out_features (int): output dimension.
        device (torch.device | None= None): Device to store the parameters on
        dtype (torch.dtype | None = None): Data type of the parameters            
    """    
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 device:torch.device|None=None, 
                 dtype:torch.dtype|None=None
                 ):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features        
        
        std = math.sqrt(2 / (self.in_features + self.out_features ))
        self.weight = nn.Parameter(torch.empty(self.out_features, 
                                               self.in_features, 
                                               device=device, 
                                               dtype=dtype
                                               )
                                   )
        torch.nn.init.trunc_normal_(self.weight, mean=0 , std=std, a=-3*std, b=3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:        
        return einsum(self.weight, x, "d_out d_in, ... d_in-> ... d_out")


class Embedding(nn.Module):
    """   
    Produces the embeddings for a batch of ids.
    Args:
        num_embeddings (int): The number of embeddings in the vocabulary
        embedding_dim (int): The size of the embedding dimensions.         
    """
    def __init__(self, 
                 num_embeddings:int,
                 embedding_dim:int, 
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None
                 ):
        
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        std = math.sqrt(2 / (self.num_embeddings + self.embedding_dim))
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, 
                                               self.embedding_dim, 
                                               device = device, 
                                               dtype = dtype
                                               )
                                   )
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, ids:torch.Tensor)-> torch.Tensor:  
        """
        Args:
            ids: Tensor[batch, sec_len]
        Returns:
            embedding vectors: Tensor[batch, sec_len, embedding dim]
        """
        assert ids is not None and ids.ndim < 3 
        if ids.ndim == 1:
            out = self.weight[ids]
        else:
            b, seq_len = ids.shape
            ids = rearrange(ids, "b seq_len-> (b seq_len)", b=b, seq_len=seq_len)
            out = self.weight[ids]
            out = rearrange(out, "(b seq_len) ...-> b seq_len ...", b=b, seq_len=seq_len)
        return out


class RMSNorm(nn.Module):
    """
    Implements layer norm for an input. The learnable gain parameters: are initialized to 1
    
    args:
        d_model: The dimensionality of RMSNorm input
        eps : Epsilon value for numerical stability
        device : Device to store the parameters on
        dtype : Data type of the parameters
    """
    def __init__(self, 
                 d_model:int, 
                 eps:float=1e-5, 
                 device:torch.device|None=None, 
                 dtype:torch.dtype|None=None
                 ):
        
        super().__init__()
        self.d_model = d_model
        self.eps = eps        
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=self.dtype)) 

    def forward(self, x:torch.Tensor)-> torch.Tensor:  
        """
        args: 
            Float[tensor,"... d_model"]: in_features to be normalized.
        returns:
            Float[tensor,"... d_model"]: normalized in_features of the same dim.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((1/self.d_model) * einsum(x, x, "... d_model, ... d_model-> ...") + self.eps)
        normalized = x / rearrange(rms, "... -> ... 1")
        normalized_x = normalized.to(in_dtype)
        # Return the result in the original dtype
        return einsum(normalized_x, self.weight, "... d_model, d_model-> ... d_model")


class FFN(nn.Module):
    """
    Implements a SwiGLU position-wise feed-forward network
    
    """
    def __init__(self, 
                 d_model:int, 
                 dff:int, 
                 device:torch.device|None=None, 
                 dtype:torch.dtype|None=None
                 ):
        
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.x1 = Linear(d_model, dff, device = device, dtype = dtype)
        self.x2 = Linear(dff, d_model, device = device, dtype = dtype)
        self.x3 = Linear(d_model, dff, device = device, dtype = dtype)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        Args: 
            x = w1_weight *SiLU(w1_weight x) ⊙ w13_weight*x)  
            w1_weight (Float[Tensor, "d_ff d_model"])
            w2_weight (Float[Tensor, "d_model d_ff"])
            w3_weight (Float[Tensor, "d_ff d_model"])
        """
        GLU_x = silu(self.x1(x)) * self.x3(x)
        return self.x2(GLU_x)


class PositionalEmbedding(nn.Module):
    """"
    Implements Rotary Positional Embedding (RoPE)
    """
    def __init__(self, 
                 d_k:int, 
                 theta:float, 
                 max_seq_len:int, 
                 device:torch.device|None=None, 
                 dtype:torch.dtype|None=None
                 ):
        
        super().__init__()        
        assert d_k % 2 ==0        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.rotation_matrix()        

    def rotation_matrix(self):
        pos_k = torch.arange(self.d_k // 2, device=self.device, dtype=self.dtype)
        pos_i = torch.arange(self.max_seq_len, device=self.device, dtype=self.dtype)
        theta_k = self.theta**(- 2 * pos_k / self.d_k)
        angles = einsum(pos_i, theta_k , "i, k-> i k")
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

    def forward(self, x:torch.Tensor, pos_ids:torch.Tensor|None=None)->torch.Tensor:              
        seq_x = x.shape[-2]
        assert seq_x <= self.max_seq_len
        if pos_ids is None:            
            sin, cos = self.sin[: seq_x], self.cos[: seq_x]
        else:                       
            assert pos_ids.ndim == 1 or pos_ids.ndim == 2       
            if  pos_ids.ndim == 1: 
                sin, cos = self.sin[pos_ids], self.cos[pos_ids]
            else:
                batch, seq = pos_ids.shape
                pos_ids = rearrange(pos_ids, "b seq-> (b seq)", b=batch, seq=seq)
                sin, cos = self.sin[pos_ids], self.cos[pos_ids]
                sin, cos = [rearrange(r, "(b seq) ...-> b seq ...", b=batch, seq=seq) for r in [sin, cos]]
                
        x1, x2 = rearrange(x, "... (chunk n)-> n ... chunk", chunk=self.d_k//2, n=2)
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos    

        stacked = torch.stack((rot_x1, rot_x2), dim=-1)    
        return rearrange(stacked, "... chunk n-> ... (chunk n)", chunk=self.d_k//2, n=2).contiguous()
        

class CausalMultiheadAttention(nn.Module):
    """
    Implements causal multi-head attention
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 theta: float, 
                 max_seq_len: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
                 ):
        
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads     
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.d_k = self.d_model // self.num_heads
        self.rope = PositionalEmbedding(self.d_k, self.theta, self.max_seq_len, device=device, dtype=dtype)
        
        self.q = Linear(self.d_model, self.d_model, device = device, dtype = dtype)
        self.k = Linear(self.d_model, self.d_model, device = device, dtype = dtype)
        self.v = Linear(self.d_model, self.d_model, device = device, dtype = dtype)
        self.o = Linear (self.d_model, self.d_model, device = device, dtype = dtype) 

    def scaled_dot_prod_attention(self, 
                                  d_k, 
                                  Q: torch.Tensor, 
                                  K: torch.Tensor, 
                                  V: torch.Tensor, 
                                  mask: torch.Tensor
                                  )-> torch.Tensor:
        """
        Calculates the attention mechanism
        """
        QK = einsum(Q, K, "... queries d_k, ... keys d_k-> ... queries keys") / math.sqrt(d_k)    
        mask = torch.where(mask, 0., -float(math.inf))
        probs = softmax (QK + mask, -1) 
        return einsum(probs, V, "... queries values, ... values d_v-> ... queries d_v")

    def forward(self, x:torch.Tensor, token_ids: torch.Tensor = None)-> torch.Tensor:
        """
        Implements multi-head attention with rotary position embedding
        """
        Q = self.q(x) 
        K = self.k(x)
        V = self.v(x)        
        d_v = self.d_k       
        q, k, v = [rearrange(i, "... seq_len (num_heads d_k)-> num_heads ... seq_len d_k", d_k = self.d_k, num_heads = self.num_heads) for i in [Q, K, V]]
        q, k = [self.rope(y, token_ids) for y in [q, k]]    
        
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device)).bool()    
        heads = self.scaled_dot_prod_attention(self.d_k, q, k, v, causal_mask)
        
        combined_heads = rearrange(heads, "num_heads ... seq_len d_k-> ... seq_len (num_heads d_k)", d_k = self.d_k, num_heads = self.num_heads).contiguous()
        return self.o(combined_heads)


class Transformer(nn.Module):
    """
    Implements the transformer module with SWiGLU FFN and RoPE 
    """
    def __init__(self, 
                 d_model: int, 
                 dff: int, 
                 num_heads: int, 
                 theta: float, 
                 max_seq_len: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
                 ):
        
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.rms_norm = RMSNorm(d_model = self.d_model, device = device, dtype = dtype)
        self.multi_head_attention = CausalMultiheadAttention(d_model=self.d_model,
                                                             num_heads=self.num_heads,
                                                             theta=self.theta,
                                                             max_seq_len=self.max_seq_len,
                                                             device=device,
                                                             dtype=dtype
                                                             )
        self.ffn = FFN(d_model=self.d_model, dff=self.dff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor = None)-> torch.Tensor:
        y = x + self.multi_head_attention(self.rms_norm(x), token_ids)
        return y + self.ffn(self.rms_norm(y))


class TransformerLM(nn.Module):
    """
    Implements the transformer language model 
    """
    def __init__(self, 
                 vocab_len:int, 
                 num_layers:int, 
                 d_model:int, 
                 dff:int, 
                 num_heads:int, 
                 theta:float, 
                 max_seq_len:int, 
                 device:torch.device|None = None, 
                 dtype:torch.dtype|None=None
                 ):
        super().__init__()
        self.vocab_len = vocab_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len        

        self.token_embedding = Embedding(self.vocab_len, self.d_model, device=device, dtype=dtype)
        self.transformer_layers = nn.ModuleList([Transformer(d_model=self.d_model,
                                                             dff=self.dff,
                                                             num_heads=self.num_heads,
                                                             theta=self.theta,
                                                             max_seq_len=self.max_seq_len,
                                                             device=device,
                                                             dtype=dtype
                                                             ) for i in range(self.num_layers)]
                                                )
        self.out_linear = Linear(self.d_model, self.vocab_len, device=device, dtype=dtype)
        self.norm = RMSNorm(self.d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor = None)-> torch.Tensor:        
        y = self.token_embedding(x)        
        for i in range(self.num_layers):
            y = self.transformer_layers[i](y)           
        
        logits = self.out_linear(self.norm(y))
        return logits        
