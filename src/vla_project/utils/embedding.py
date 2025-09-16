import torch
from jaxtyping import Float, Int
from torch import nn


class RoPEEmbedding(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None,
    ):
        super().__init__()
        self.dim = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        # theta_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)).to(
            device=device,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute all cos and sin values
        self._set_cos_sin_cache(seq_len=max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        # t 的形状是 (seq_len,)
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,  # type: ignore  # noqa: PGH003
            dtype=self.inv_freq.dtype,  # type: ignore  # noqa: PGH003
        )  # type: ignore  # noqa: PGH003

        # outter product
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # emb 的形状是 (seq_len, dim)
        # 将 freqs 扩展一倍，维度变为 (seq_len, dim),奇偶维度相同
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        # cache cos and sin values.
        # shape is (1, seq_len, 1, dim)
        # self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        # self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, " ... seq_len d_k"],
        token_positions: Int[torch.Tensor, " ... seq_len"],
    ) -> Float[torch.Tensor, " ... seq_len d_k"]:
        seq_len = x.shape[-2]

        cos: torch.Tensor = self.cos_cached[:seq_len, :]  # type: ignore  # noqa: PGH003
        sin: torch.Tensor = self.sin_cached[:seq_len, :]  # type: ignore  # noqa: PGH003
        cos = cos[token_positions, :]
        sin = sin[token_positions, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # 将 x1 变号，然后和 x2 拼接, 实现旋转效果
        # change the sign of x
        return torch.reshape(torch.stack((-x2, x1), dim=-1), x.shape)
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # # 将后半部分取反后与前半部分拼接
        # return torch.cat((-x2, x1), dim=-1)
        # return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000,
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [B, L, H, D], where B is batch size,
            L is sequence length, H is number of heads, and D is head dimension.
        positions (torch.Tensor): Tensor of positions with shape [B, L].
        max_wavelength (float, optional): The maximum wavelength for the sinusoidal
            embeddings. Defaults to 10_000.

    Returns:
        torch.Tensor: The tensor with RoPE applied, having the same shape as the input.

    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


if __name__ == "__main__":
    from einops import rearrange
    rope = RoPEEmbedding(theta=10000, d_k=64, max_seq_len=128, device=torch.device("cpu"))
    x = torch.randn(2, 128, 64)
    positions = torch.arange(0, 128).unsqueeze(0).repeat(2, 1)
    y = rope(x, positions)
    print(y.shape)
    x2 = rearrange(x, "b s (h l) -> b s h l", h=4)
    # x2 = torch.reshape(x, (2, 128, 4, 16))
    y2 = apply_rope(x2, positions)
    y2 = rearrange(y2, "b s h l -> b s (h l)")
    print(y2.shape)
    print(torch.allclose(y, y2, atol=1e-5))

    print(y2)

    print(y)