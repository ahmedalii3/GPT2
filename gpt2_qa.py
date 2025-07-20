import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import inspect  # Add this line
from dataclasses import dataclass
from datasets import load_dataset
# Assuming the CausalSelfAttention, MLP, Block, and GPTConfig classes remain unchanged
# (omitted here for brevity but should be included from the original code)
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):
    # (Original GPT class code remains unchanged, included for completeness)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            mask = (targets != 0).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.contiguous().view(-1), 
                                 reduction='none')
            loss = (loss * mask.view(-1)).sum() / mask.sum()
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# Dataset preparation (example)
def prepare_qa_dataset():
    dataset = load_dataset("squad", split="train")
    qa_pairs = []
    for item in dataset:
        question = item["question"]
        answer = item["answers"]["text"][0]
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

# Data loader for question-answer pairs
from torch.nn.utils.rnn import pad_sequence



    # ... (other methods unchanged)

def prepare_qa_dataset():
    dataset = load_dataset("squad", split="train")
    qa_pairs = []
    for item in dataset:
        question = item["question"]
        answer = item["answers"]["text"][0]
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

class QADataset:
    def __init__(self, qa_pairs, enc, block_size):
        self.qa_pairs = qa_pairs
        self.enc = enc
        self.block_size = block_size

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        text = f"Question: {qa['question']} Answer: {qa['answer']} <|endoftext|>"
        tokens = self.enc.encode(text, allowed_special={'<|endoftext|>'})
        tokens = tokens[:self.block_size - 1]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

def collate_fn(batch):
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Fine-tuning function
def fine_tune_model(model, dataset, device, epochs=4, batch_size=4, learning_rate=3e-5):
    model.train()
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device_type=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")
# Inference function for question answering
def answer_question(model, enc, question, device, max_length=50):
    model.eval()
    prompt = f"Question: {question} Answer:"
    tokens = torch.tensor(enc.encode(prompt, allowed_special={'<|endoftext|>'}), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        while tokens.size(1) < max_length:
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim=1)
            if xcol.item() == enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]:
                break
    
    generated = enc.decode(tokens[0].tolist())
    try:
        answer = generated.split("Answer:")[1].strip()
        answer = answer.replace("<|endoftext|>", "").strip()
    except IndexError:
        answer = "Could not generate a valid answer."
    return answer
# Main execution
if __name__ == "__main__":
    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("Warning: MPS backend is not available. Using CPU.")
    
    # Load pretrained model
    model = GPT.from_pretrained("gpt2")
    model.to(device)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Prepare dataset
    qa_pairs = prepare_qa_dataset()
    dataset = QADataset(qa_pairs, enc, block_size=1024)
    
    # Fine-tune the model
    fine_tune_model(model, dataset, device, epochs=3, batch_size=2, learning_rate=3e-5)
    
    # Test the model
    test_questions = [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?",
        "What is My name?"
    ]
    
    for question in test_questions:
        answer = answer_question(model, enc, question, device)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")