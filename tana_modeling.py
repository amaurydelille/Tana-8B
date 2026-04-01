import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
from transformers import AutoTokenizer

MAX_LEN = 2048

class Tokenizer:
    def __init__(self, tokenizer_id: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN - 1,
        )
        return enc["input_ids"].to(device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            token_ids.detach().cpu().tolist(),
            skip_special_tokens=skip_special_tokens,
        )

TOKENIZER_ID = "mistralai/Mistral-7B-v0.1"
TOKENIZER = Tokenizer(TOKENIZER_ID)

class RoPE:
    @staticmethod
    def build_rope_cache(max_seq_length: int, head_dim: int, device: Literal["cpu", "cuda", "mps"]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert head_dim % 2 == 0, "RoPE head dimension must be even"

        half_dim = head_dim // 2

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, device=device) / half_dim)
        )

        positions = torch.arange(max_seq_length, device=device)

        angles = torch.einsum("i,j->ij", positions, inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        assert d % 2 == 0, "RoPE dimension must be even"

        x = x.view(b, h, s, d // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        cos = cos[:s].unsqueeze(0).unsqueeze(0)
        sin = sin[:s].unsqueeze(0).unsqueeze(0)

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.stack([out1, out2], dim=-1).reshape(b, h, s, d)

class SwiGLU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int) -> None:
        super().__init__()
        self.linear_proj_1 = nn.Linear(input_features, hidden_features)
        self.linear_proj_2 = nn.Linear(input_features, hidden_features)
        self.down_proj = nn.Linear(hidden_features, input_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.linear_proj_1(x)) * self.linear_proj_2(x)
        return self.down_proj(h)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device: Literal["cpu", "cuda", "mps"]) -> None:
        assert d_model % n_heads == 0, "D model should be divisible by number of heads."
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Parameter(torch.randn(d_model, d_model))
        self.k_proj = nn.Parameter(torch.rand(d_model, d_model))
        self.v_proj = nn.Parameter(torch.rand(d_model, d_model))
        
        self.out_proj = nn.Parameter(torch.rand(d_model, d_model))

        cos, sin = RoPE.build_rope_cache(
            max_seq_length=MAX_LEN,
            head_dim=self.d_head,
            device=device,
        )

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self._init_tensors()

    def _init_tensors(self) -> None:
        torch.nn.init.xavier_normal_(self.q_proj)
        torch.nn.init.xavier_normal_(self.k_proj)
        torch.nn.init.xavier_normal_(self.v_proj)
        torch.nn.init.xavier_normal_(self.out_proj)

    def forward(self, x: torch.Tensor, mask: bool = True) -> torch.Tensor:
        B, N, D = x.shape
        q = (x @ self.q_proj).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = (x @ self.k_proj).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = (x @ self.v_proj).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
 
        q = RoPE.apply_rope(q, self.cos, self.sin)
        k = RoPE.apply_rope(k, self.cos, self.sin)

        attention_scores = (q @ k.transpose(-2, -1)) / (math.sqrt(self.d_head))

        if mask:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device))
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))

        attention_values = F.softmax(attention_scores, dim=-1) @ v
        attention_values = attention_values.transpose(1, 2).contiguous().view(B, N, self.d_model)

        return attention_values @ self.out_proj

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_hidden) for _ in range(n_experts)
        ])
        self.top_k = top_k
        self.gate_proj = nn.Linear(d_model, n_experts)
        self.n_experts = n_experts

    def _load_balance_loss(self, router_logits: torch.Tensor, top_k_indices: torch.Tensor, n_experts: int) -> torch.Tensor:
            gate_probs = F.softmax(router_logits, dim=-1)
            gate_probs_flat = gate_probs.view(-1, n_experts)
            N = gate_probs_flat.size(0)
            importance = gate_probs_flat.sum(dim=0) / N
            indices_flat = top_k_indices.view(-1)
            load = torch.bincount(indices_flat, minlength=n_experts).float()
            load = load / indices_flat.numel()
            return (importance * load).sum() * n_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate_proj(x)
        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x)
        for i in range(self.top_k):
            exprt_idx = top_k_indices[:, :, i]
            weight = top_k_probs[:, :, i].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (exprt_idx == e).unsqueeze(-1)
                if mask.any():
                    expert_out = self.experts[e](x)
                    output = output + mask.float() * weight * expert_out

        auxiliary_loss = self._load_balance_loss(router_logits, top_k_indices, self.n_experts)

        return output, auxiliary_loss

class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: int, n_experts: int, top_k: int, vocab_size: int, device: Literal["cpu", "cuda", "mps"]) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model, device=device)
        self.pre_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, device=device)
        self.experts = MixtureOfExperts(d_model, d_hidden, n_experts, top_k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype in (torch.long, torch.int32, torch.int64):
            x = self.embeddings(x)
        residual = x
        pre_norm = self.pre_norm(x)
        attention_output = self.attention(pre_norm) + residual
        norm_output = self.norm(attention_output)
        experts_output, auxiliary_loss = self.experts(norm_output)

        return experts_output, auxiliary_loss

class Tana(nn.Module):
    def __init__(self, n_decoders: int, d_model: int, n_heads: int, d_hidden: int, n_experts: int, top_k: int, vocab_size: int, device: Literal["cpu", "cuda", "mps"]) -> None:
        super().__init__()
        self.decoders = nn.ModuleList([
            Decoder(d_model, n_heads, d_hidden, n_experts, top_k, vocab_size, device) for _ in range(n_decoders)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, device=device, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_aux = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for decoder in self.decoders:
            x, aux = decoder(x)
            total_aux = total_aux + aux.float()
        return self.lm_head(x), total_aux

    @torch.inference_mode()
    def generate(
        self,
        tokenizer: Tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        max_length: Optional[int] = None,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        cap = min(max_length or MAX_LEN, MAX_LEN)
        input_ids = tokenizer.encode(prompt, device)
        eos_id = tokenizer.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= cap:
                break
            logits, _ = self(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_id is not None and next_token.item() == eos_id:
                break
        return tokenizer.decode(input_ids[0])