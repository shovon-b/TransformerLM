import os
import math
import numpy as np
import torch
import torch.nn as nn
import argparse
import wandb
from jaxtyping import Float, Int
from einops import einsum, rearrange
from layers import *
from utils import *
from optimizer import AdamW



parser = argparse.ArgumentParser()
#model_inputs
parser.add_argument("--vocab_len", type=int, default = 10000,
                    help="Vocabulary length")
parser.add_argument("--num_layers", type=int, default=4,
                    help="Number of transformer layers")
parser.add_argument("--d_model", type=int, default=512,
                    help="The model's embdedding dimension")
parser.add_argument("--dff", type=int, default=1334,
                    help="Width of FFN, should be (8/3)*d_model and dff % 64 ==0")
parser.add_argument("--num_heads", type=int, default=16,
                    help="Number of heads for self.attention")
parser.add_argument("--theta", type=float, default=1000.0,
                    help="angle for RoPE")
parser.add_argument("--context_len", type=int, default = 256,
                    help="Context length for RoPE")
#data
parser.add_argument("--train_filename", type=str, default= "TinyStories_train_tokens.npy",
                    help="Name of the tokenized training dataset, assumed to be in /data/")
parser.add_argument("--validation_filename", type=str, default= "TinyStories_valid_tokens.npy",
                    help="Name of the tokenized validation dataset, assumed to be in /data/")
parser.add_argument("--seq_len", type=int, default = 256,
                    help="Sequence_length of the training data")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size of training data")
parser.add_argument("--total_step", type=int, default=10000,
                    help="total step number for training")

#optimization
parser.add_argument("--lr", type=float, default=.001,
                    help="learning rate")
parser.add_argument("--beta_1", type=float, default=.9,
                    help="betas_1")
parser.add_argument("--beta_2", type=float, default=.995,
                    help="betas_2")
parser.add_argument("--weight_decay", type=float, default=.1,
                    help="weight decay")
#training
parser.add_argument("--wandb", type=str, default=None,
                    help="wandb will be used as the run_name within wandb")
parser.add_argument("--save_interval", type=int, default=100,
                   help="interval to save the model")
parser.add_argument("--save_model_name", type=str, default="model",
                   help="model_name to save the model")
parser.add_argument("--validation_interval", type=int, default=20,
                   help="interval to validate the model")
parser.add_argument("--load_model", type=str, default=None,
                   help="model_name.pt to reload model from last training point")

args = parser.parse_args()

model = TransformerLM(vocab_len=args.vocab_len, 
                      num_layers=args.num_layers, 
                      d_model=args.d_model, 
                      dff=args.dff, 
                      num_heads=args.num_heads, 
                      theta=args.theta, 
                      max_seq_len=args.context_len                     
                     )

optim = AdamW(params=model.parameters(), 
                        lr=args.lr, 
                        betas=(args.beta_1,args.beta_2), 
                        weight_decay=args.weight_decay
                       )

if  args.load_model is not None:
    iteration = load_checkpoint(args.load_model, model, optim)
else:
    iteration = 0


device = "cuda" if torch.cuda.is_available() else "cpu"    
model.to(device)

run = None
if args.wandb is not None:    
    run = wandb.init(
    entity="your entity",
    project="Transformer_LM", 
    name=args.wandb)
    wandb.config.update(args)
    

for it in range(iteration, args.total_step):
    model.train()
    dir_name = "data"
    input_data_path = os.path.join(dir_name, args.train_filename)
    dataset= np.load(input_data_path, mmap_mode='r')
    inputs, preds_gt = get_batch(dataset=dataset, 
                       batch_size=args.batch_size, 
                       context_length=args.seq_len, 
                       device=device)
    
    logits = model(inputs)
    loss =  cross_entropy_loss(logits, preds_gt)

    if args.wandb is not None:
        run.log({"Train": loss.item()}, step=it)
    
    optim.zero_grad() 
    loss.backward() 
    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    lr = cosine_lr_schedule(it=it,
                            max_learning_rate=args.lr,
                            min_learning_rate=0.1 * args.lr,
                            warmup_iters=int(0.2*args.total_step),
                            cosine_cycle_iters=args.total_step)    
    for group in optim.param_groups: 
        group["lr"] = lr
    optim.step()

    if (it + 1) % args.save_interval == 0:
        save_checkpoint(model=model, 
                        optimizer=optim,
                        iteration=it, 
                        out_filename=args.save_model_name)
    
    if (it + 1) % args.validation_interval == 0:
        model.eval()
        dir_name = "data"
        valid_data_path = os.path.join(dir_name, args.validation_filename)
        valid_dataset= np.load(valid_data_path, mmap_mode='r')        
        with torch.no_grad():
            val_inputs, val_preds_gt = get_batch(dataset=valid_dataset, 
                           batch_size=args.batch_size, 
                           context_length=args.seq_len, 
                           device=device)
        
            val_logits = model(val_inputs)
            val_loss =  cross_entropy_loss(val_logits, val_preds_gt)
            if args.wandb is not None:
                run.log({"Validation": val_loss.item()}, step=it)
        
    

if args.wandb is not None:
    run.finish()

