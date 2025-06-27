import os
import torch
import torch.nn as nn
import argparse
from tokenization import *

from jaxtyping import Float, Int
from einops import einsum, rearrange
from layers import *
from utils import softmax

parser = argparse.ArgumentParser()

parser.add_argument("--load_model", type=str, default="model.pt",
                   help="model_name.pt to reload model from last training point in /models/")
parser.add_argument("--prompt", type=str, default="A brown fox",
                   help="User prompt to generate text")
parser.add_argument("--temp", type=float, default= 0.00001,
                   help="Temperature hyperparameter")
parser.add_argument("--max_token", type=int, default= 256,
                   help="Max token num to generate")

args = parser.parse_args()

#ensure that the model parameters are the same as in training
model = TransformerLM(vocab_len=10000, 
                      num_layers=4, 
                      d_model=512, 
                      dff=1334, 
                      num_heads=16, 
                      theta=1000.0, 
                      max_seq_len=256                    
                     )

max_seq_len = model.max_seq_len
device = "cuda" if torch.cuda.is_available() else "cpu"    
model.to(device)

dir_name= "models"
src_path = os.path.join(dir_name, args.load_model)
if not os.path.exists(src_path):
    print("model not found.")
else:
    saved_checkpoints = torch.load(src_path)
    model.load_state_dict(saved_checkpoints["model"])
    print("Model loaded.")
    
#preparing the prompt and the special token
special_token = "<|endoftext|>"
vocab_filepath = os.path.join("data", "TinyStories_vocab.pkl")
merges_filepath = os.path.join("data", "TinyStories_merges.pkl")
z = Tokenizer.from_files(vocab_filepath,merges_filepath, [special_token])
prompt_tokens = z.encode(args.prompt)
special_token_byte = special_token.encode("utf-8", errors="ignore")
end_token = z.reverse_vocab[special_token_byte]

prompt = torch.tensor(prompt_tokens , dtype=torch.long, device=device)
if prompt.dim() == 1: prompt = prompt.unsqueeze(0)

model_input = prompt
with torch.no_grad():    
    for _ in range(args.max_token):
        if model_input.shape[-1] > max_seq_len:
            model_input = prompt[:, -max_seq_len :]
        logits = model(model_input)
        next_logit = logits[:, -1]
        probs = softmax(next_logit / args.temp, -1)
        next_token = torch.multinomial(probs, 1)
        prompt = torch.cat((prompt,next_token), dim=-1)        
        if next_token.item() == end_token:
            break
        model_input = prompt
    completed_prompt_ids = prompt.squeeze(0).cpu().numpy()

print(z.decode(completed_prompt_ids))
