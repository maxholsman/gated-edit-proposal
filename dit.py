import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- helpers ----------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t_scalar):
        # t_scalar expected in [0, 1] (i.e., t/T)
        t_freq = self.timestep_embedding(t_scalar, self.frequency_embedding_size)
        return self.mlp(t_freq)

class LengthEmbedder(nn.Module):
    """Same parametrization as timestep MLP, used for target length L."""
    def __init__(self, hidden_size, frequency_embedding_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def sinusoidal(x, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=x.device) / half
        )
        args = x[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, L_scalar):
        # L can be raw length or normalized (e.g., L/L_max). Either way works consistently.
        e = self.sinusoidal(L_scalar, self.frequency_embedding_size)
        return self.mlp(e)

# ---------- RoPE (rotary) ----------
class RotaryEmbedding(nn.Module):
    """
    Standard RoPE for 1D sequences.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [L, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)            # [L, dim]
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]  # [1,1,L,dim]

    @staticmethod
    def apply_rotary(x, cos, sin):
        # x: [B, nH, L, Hd], cos/sin: [1,1,L,Hd]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)

class RopeAttention(nn.Module):
    """
    MHA with RoPE on q,k. Slightly mirrors timm Attention but supports masks.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Hd]

        cos, sin = self.rope.get_cos_sin(L, x.device)  # [1,1,L,Hd]
        q = RotaryEmbedding.apply_rotary(q, cos, sin)
        k = RotaryEmbedding.apply_rotary(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1,2).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ---------- DiT blocks (unchanged structure, new Attention) ----------
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **_):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = RopeAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalTokenLayer(nn.Module):
    """
    Final adaLN + linear head to per-token vocab logits.
    """
    def __init__(self, hidden_size, vocab_size_plus1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, vocab_size_plus1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)  # [B, L, D]
        logits = self.linear(x)                         # [B, L, V+1]
        return logits

# ---------- Sequence DiT ----------
class SequenceDiT(nn.Module):
    """
    DiT for sequences (protein tokens).
    Inputs:
      - x_tokens: (B, L, D_esm) precomputed ESM-2 embeddings (fixed; we only project)
      - L_target: (B,) scalar length for each sequence (int or float)
      - t: (B,) current timestep (int), T: scalar total timesteps (int)
    Output:
      - logits q: (B, L, V+1)
    """
    def __init__(
        self,
        d_esm,                   # input dim from ESM-2
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        vocab_size_plus1=26,     # 20 AA + special tokens; set yours (V+1)
    ):
        super().__init__()
        self.proj_in = nn.Linear(d_esm, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.L_embedder = LengthEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final = FinalTokenLayer(hidden_size, vocab_size_plus1)

        # init similar to DiT
        self.apply(self._basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.L_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.L_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

    @staticmethod
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x_tokens, L_target, t, T, attn_mask=None):
        """
        x_tokens: (B, L, D_esm)
        L_target: (B,) e.g., target length (int). You may pass raw value or normalized.
        t: (B,) current step; T: int total steps
        attn_mask: optional (B, 1, L, L) or (B, L) -> will be broadcast to attention
        """
        B, L, _ = x_tokens.shape

        x = self.proj_in(x_tokens)  # (B, L, D)
        # RoPE handles positions; no additive 1D pos-emb needed.

        # Conditioning vector c (B, D)
        t_norm = t.float() / float(T)   # normalize to [0,1]
        t_emb = self.t_embedder(t_norm) # (B, D)
        # length embedding: raw or normalized (you can pass L_target / L_max)
        L_emb = self.L_embedder(L_target.float())  # (B, D)
        c = t_emb + L_emb

        # Optional mask handling: accept (B,L) and convert to attn mask if needed
        if attn_mask is not None and attn_mask.dim() == 2:
            # attn_mask: 1 for keep, 0 for pad
            am = attn_mask[:, None, None, :].to(x.dtype)   # (B,1,1,L)
            attn_mask = (am @ am.transpose(-2, -1)).bool() # (B,1,L,L)

        for blk in self.blocks:
            x = blk(x, c, attn_mask=attn_mask)  # (B, L, D)

        logits = self.final(x, c)  # (B, L, V+1)
        return logits
