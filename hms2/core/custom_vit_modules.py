import torch
import torch.nn as nn
from einops import rearrange, repeat


def _posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    if dim % 4 != 0:
        raise ValueError('feature dimension must be multiple of 4 for sincos emb')
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _Attention(dim, heads=heads, dim_head=dim_head),
                        _FeedForward(dim, mlp_dim),
                    ],
                ),
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, channels, dim_head=64):
        super().__init__()
        self.to_embedding = nn.Linear(in_features=channels, out_features=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = _Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, input):
        x = rearrange(input, 'b d ... -> b ... d')  # [N, H, W, C]
        x = self.to_embedding(x)  # [N, H, W, D]

        pe = _posemb_sincos_2d(x)  # [N, H * W, D]
        x = rearrange(x, 'b ... d -> b (...) d')  # [N, H * W, D]
        x = x + pe  # [N, H * W, D]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])  # [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [N, H * W + 1, D]

        x = self.transformer(x)
        x = x[:, 0]

        return self.linear_head(x)
