import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
from dataclasses import dataclass
import time
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'



@dataclass
class ModelConfig:
    vocab_size: int = 50304
    embd_dim: int = 256
    num_head: int = 8
    latent_dim: int = 64
    rope_dim: int = 64
    d_ff: int = 512
    num_experts: int = 8
    top_k: int = 2
    load_balance_alpha: float = 0.01
    capacity_factor: float = 1.25
    bias_update_speed: float = 0.1
    load_tracking_momentum: float = 0.1
    num_layers: int = 4
    dropout: float = 0.1



class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.num_head = config.num_head
        self.embd_dim = config.embd_dim
        self.head_dim = config.embd_dim // config.num_head
        self.rope_dim = min(config.rope_dim, self.head_dim)
        self.rope_half = self.rope_dim // 2
        self.scale = self.head_dim ** -0.5
        self.compress_kv = nn.Linear(config.embd_dim, config.latent_dim, bias=False)
        self.kv_proj = nn.Linear(config.latent_dim, 2 * config.embd_dim, bias=False)
        self.compress_q = nn.Linear(config.embd_dim, config.latent_dim, bias=False)
        self.q_proj = nn.Linear(config.latent_dim, config.embd_dim, bias=False)
        self.q_rope = nn.Linear(config.latent_dim, config.num_head * self.rope_dim, bias=False)
        self.k_rope = nn.Linear(config.latent_dim, self.rope_dim, bias=False)
        self.out_proj = nn.Linear(config.embd_dim, config.embd_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cos_cache = {}
        self._sin_cache = {}

    def forward(self, x):
        B, S, _ = x.shape
        c_kv = self.compress_kv(x)
        kv = self.kv_proj(c_kv)
        c_q = self.compress_q(x)
        q_c = self.q_proj(c_q)

        q_r = self.q_rope(c_q).view(B, S, self.num_head, self.rope_dim).transpose(1, 2)
        k_r = self.k_rope(c_kv).unsqueeze(1).expand(-1, self.num_head, -1, -1)
        q_c = q_c.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        k_c, v = kv.chunk(2, dim=-1)

        k_c = k_c.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        q_r = self._apply_rope(q_r, S)
        k_r = self._apply_rope(k_r, S)

        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r], dim=-1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = attended.transpose(1, 2).contiguous().view(B, S, self.embd_dim)
        out = self.out_proj(out)
        return self.attn_dropout(out)
    
    def _get_rope_cache(self, S, device):
        if S in self._cos_cache:
            return self._cos_cache[S], self._sin_cache[S]
        
        pos = torch.arange(S, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, self.inv_freq)
        cos_vals = torch.cos(freqs).view(1, 1, S, self.rope_half)
        sin_vals = torch.sin(freqs).view(1, 1, S, self.rope_half)
        self._cos_cache[S], self._sin_cache[S] = cos_vals, sin_vals
        return cos_vals, sin_vals
    
    def _apply_rope(self, x, S):
        cos_vals, sin_vals = self._get_rope_cache(S, x.device)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rot = torch.empty_like(x[..., :self.rope_dim])
        x_rot[..., 0::2] = x_even * cos_vals - x_odd * sin_vals
        x_rot[..., 1::2] = x_even * sin_vals + x_odd * cos_vals
        return x_rot
    




class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.embd_dim, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.embd_dim, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.embd_dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.ffn_dropout(out)
    
class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.bias_update_speed     = config.bias_update_speed
        self.load_tracking_momentum = config.load_tracking_momentum
        self.experts      = nn.ModuleList([FFN(config) for _ in range(self.num_experts)])
        self.centroids    = nn.Parameter(torch.randn(self.num_experts, config.embd_dim))
        self.routing_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        self.register_buffer("expert_load_ema", torch.zeros(self.num_experts))

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.view(-1, H)
        affinity = torch.sigmoid(x_flat @ self.centroids.T)
        biased = affinity + self.routing_bias
        topk_scores, topk_idx = torch.topk(biased, self.top_k, dim=-1)
        topk_w = F.softmax(topk_scores, dim=-1)
        token_idx   = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.top_k)
        flat_idx    = topk_idx.flatten()
        flat_w      = topk_w.flatten()
        order       = torch.argsort(flat_idx)
        perm_tokens = x_flat[token_idx[order]]
        perm_w      = flat_w[order]
        counts = torch.bincount(flat_idx, minlength=self.num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        expert_outs = torch.empty_like(perm_tokens)

        for i in range(self.num_experts):
            c = counts[i].item()
            if c > 0:
                s = starts[i]
                expert_outs[s:s+c] = self.experts[i](perm_tokens[s:s+c])

        weighted = expert_outs * perm_w.unsqueeze(1)
        out_flat = torch.zeros_like(x_flat)
        out_flat.scatter_add_(0, token_idx[order].unsqueeze(1).expand_as(weighted), weighted)
        return out_flat.view(B, S, H), counts.detach()
    

def update_moe_bias(moe_layer: MOELayer, expert_counts: torch.Tensor):
    total = expert_counts.sum().item()

    if total == 0:
        return
    load = expert_counts.float() / total
    expected = 1.0 / moe_layer.num_experts

    moe_layer.expert_load_ema.mul_(1 - moe_layer.load_tracking_momentum) #type: ignore
    moe_layer.expert_load_ema.add_(moe_layer.load_tracking_momentum * load) # type: ignore
    delta = moe_layer.bias_update_speed * (expected - moe_layer.expert_load_ema)
    moe_layer.routing_bias.add_(delta)




class DeepseekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1  = nn.LayerNorm(config.embd_dim)
        self.attn = MLA(config)
        self.ln2  = nn.LayerNorm(config.embd_dim)
        self.ffn  = MOELayer(config)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        ffn_out, counts = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        update_moe_bias(self.ffn, counts)
        return x, counts
    


class DeepseekModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.token_dropout = nn.Dropout(config.dropout)
        self.blocks     = nn.ModuleList([DeepseekBlock(config) for _ in range(config.num_layers)])
        self.ln_final   = nn.LayerNorm(config.embd_dim)
        self.head       = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

        self.head.weight = self.token_embd.weight

        self.apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, input_ids):
        x = self.token_dropout(self.token_embd(input_ids))
        all_counts = []
        for block in self.blocks:
            x, counts = block(x)
            all_counts.append(counts)
        x = self.ln_final(x)
        return self.head(x), all_counts


config = ModelConfig()
model = DeepseekModel(config)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"{total_parameters / 1e6:.2f}M")


def load_tokens(file_path):
    enc = tiktoken.get_encoding("gpt2")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    tokens = enc.encode(data)
    return torch.tensor(tokens, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, file_path, split='train', train_ratio=0.9):
        self.B = B
        self.T = T
        self.split = split
        self.train_ratio = train_ratio
        self.tokens = load_tokens(file_path)
        self.total_tokens = len(self.tokens)
        split_index = int(self.total_tokens * train_ratio)
        if split == 'train':
            self.tokens = self.tokens[:split_index]
        elif split == 'val':
            self.tokens = self.tokens[split_index:]
        else:
            raise ValueError("split must be either 'train' or 'val'")
        self.reset()

    def reset(self):
        self.current_position = 0
    def next_batch(self):
        B, T = self.B, self.T
        end_position = self.current_position + B * T + 1
        if end_position > len(self.tokens):
            self.reset()
            end_position = self.current_position + B * T + 1
            if end_position > len(self.tokens):
                raise ValueError("Batch size and sequence length exceed data length.")
        buf = self.tokens[self.current_position:end_position]
        x = (buf[:-1]).view(B, T)  # Inputs
        y = (buf[1:]).view(B, T)   # Targets
        self.current_position += B * T
        if self.current_position + B * T > len(self.tokens):
            self.reset()
        return x, y
    


if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")
total_batch_size =  65536
B = 64 # micro batch size
T = 256 # sequence length
file_path = 'data.txt'
assert total_batch_size % (B * T ) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)     # grad accum steps
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
train_loader = DataLoaderLite(B, T, file_path, split='train')
val_loader = DataLoaderLite(B, T, file_path, split='val')


max_lr = 1e-3
min_lr = max_lr * 0.01
warmup_steps = 2000
max_steps = 5000
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)




optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
model.to(device)


for step in range(max_steps):

    t0 = time.time()
    # checking val loss after every 200 steps
    if step % 200 == 0:
        model.eval()
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            _, y = val_loader.next_batch()
            y = y.to(device)
            x = y[:, :-1]
            targets = y[:, 1:]
            with torch.no_grad():
                logits, _ = model(x)
                lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss_accum += lm_loss.item() / val_loss_steps
        print(f"Validation loss: {val_loss_accum:.4f}")

    # checking gen tokens after every 100 steps
    if step % 100 == 0:
        model.eval()
        prompt = torch.tensor(enc.encode("love"), dtype=torch.long).unsqueeze(0).to(device)
        xgen = prompt.repeat(5, 1)
        while xgen.size(1) < 32:
            with torch.no_grad():
                logits, _ = model(xgen)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, next_token), dim=1)
        for i in range(5):
            print(f"sample {i}: {enc.decode(xgen[i].tolist())}")

    # training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        _, y = train_loader.next_batch()
        y = y.to(device)
        x = y[:, :-1]
        targets = y[:, 1:]
        logits, all_expert_counts = model(x)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = lm_loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        for block, expert_counts in zip(model.blocks, all_expert_counts):
            update_moe_bias(block.ffn, expert_counts)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / (dt / 1000)
    print(f"step:{step} | loss: {loss_accum.item():.4f} | lr {lr:.4e} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
