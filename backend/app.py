import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
import inspect
from dataclasses import dataclass
from flask import Flask, request, jsonify
import logging
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# Define GPTConfig, CausalSelfAttention, MLP, Block, and GPT classes
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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

class GPT(nn.Module):
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

# Initialize model, tokenizer, and device
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "cpu":
    print("Warning: MPS not available. Using CPU.")
model = GPT.from_pretrained("gpt2")
enc = tiktoken.get_encoding("gpt2")
END_OF_TEXT = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"} )[0]

# Load fine-tuned weights
weights_path = "/Users/ahmed_ali/Downloads/gpt2_finetuned_qa_final.pth"  # Updated to match your path
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded fine-tuned model weights from {weights_path}")
except Exception as e:
    logger.error(f"Error loading weights: {e}")
    raise Exception(f"Failed to load weights from {weights_path}: {e}")

# def nucleus_sampling(probs, p=0.3):
#     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
#     mask = cumulative_probs <= p
#     top_p_probs = sorted_probs * mask.float()
#     prob_sum = top_p_probs.sum(dim=-1, keepdim=True)
#     # Handle case where prob_sum is zero
#     if prob_sum.item() == 0:
#         # Fallback to uniform sampling over all tokens
#         top_p_probs = torch.ones_like(probs) / probs.size(-1)
#     else:
#         top_p_probs = top_p_probs / prob_sum
#     try:
#         ix = torch.multinomial(top_p_probs, 1)
#         return torch.gather(sorted_indices, -1, ix)
#     except Exception as e:
#         logger.error(f"Error in nucleus sampling: {e}")
#         # Fallback to greedy sampling
#         return torch.argmax(probs, dim=-1, keepdim=True)

# def answer_question(model, enc, question, device, max_length=100):
#     try:
#         if "my name" in question.lower():
#             return "I don't have access to your name. Please provide more context."

#         model.eval()
#         prompt = f"Question: {question} Answer:"
#         tokens = torch.tensor(enc.encode(prompt, allowed_special={'<|endoftext|>'}), dtype=torch.long).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             while tokens.size(1) < max_length:
#                 logits, _ = model(tokens)
#                 logits = logits[:, -1, :]
#                 # Clip logits to prevent overflow
#                 logits = torch.clamp(logits, -100, 100)
#                 probs = F.softmax(logits, dim=-1)
#                 ix = nucleus_sampling(probs, p=0.7)
#                 xcol = ix
#                 tokens = torch.cat((tokens, xcol), dim=1)
#                 if xcol.item() == enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]:
#                     break
        
#         generated = enc.decode(tokens[0].tolist())
#         try:
#             answer = generated.split("Answer:")[1].strip()
#             answer = answer.replace("<|endoftext|>", "").strip()
#         except IndexError:
#             answer = "Could not generate a valid answer."
#         return answer

#     except Exception as e:
#         logger.error(f"Error in answer_question: {e}")
#         return f"Error generating answer: {str(e)}"
def nucleus_sampling(probs, p=0.7):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= p
    if not torch.any(mask):
        # Fallback to top-1 sampling (greedy)
        return torch.argmax(probs, dim=-1, keepdim=True)
    top_p_probs = sorted_probs * mask.float()
    prob_sum = top_p_probs.sum(dim=-1, keepdim=True)
    top_p_probs = top_p_probs / prob_sum
    ix = torch.multinomial(top_p_probs, 1)
    return torch.gather(sorted_indices, -1, ix)

# --------- Answer Question Logic ----------
def answer_question(model, enc, question, device, max_length=100):
    try:
        if "my name" in question.lower():
            return "I don't have access to your name. Please provide more context."

        model.eval()
        prompt = f"Question: {question} Answer:"
        tokens = torch.tensor(enc.encode(prompt, allowed_special={'<|endoftext|>'}), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            while tokens.size(1) < max_length:
                logits, _ = model(tokens)
                logits = logits[:, -1, :]
                logits = torch.clamp(logits, -100, 100)
                probs = F.softmax(logits, dim=-1)
                ix = nucleus_sampling(probs, p=0.7)
                tokens = torch.cat((tokens, ix), dim=1)
                if ix.item() == END_OF_TEXT:
                    break

        decoded = enc.decode(tokens[0].tolist())
        answer = decoded.split("Answer:")[1].split("<|endoftext|>")[0].strip()
        return answer if answer else "No answer generated."

    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return "Error generating answer."


@app.route("/answer", methods=["POST"])
def answer_question_route():
    print("✅ /answer endpoint was hit with POST")
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "Question is required"}), 400

        answer = answer_question(model, enc, question, device)
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Test questions for direct testing
    test_questions = [
        "What is the capital of France?",
        "Who wrote Pride and Prejudice?",
        "What is my name?"
    ]
    for question in test_questions:
        answer = answer_question(model, enc, question, device)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
    app.run(host="0.0.0.0", port=5001)