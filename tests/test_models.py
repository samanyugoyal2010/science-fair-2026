import torch

from src.models.hybrid_splitstream import HybridConfig, SlidingWindowAttention, SplitStreamHybridLM
from src.models.kan import KANConfig, KANLinear
from src.models.ssm_block import SSMConfig, SimpleSelectiveSSM
from src.models.transformer_baseline import GPTBaseline, TransformerConfig
from src.train.train import count_params


def test_sliding_window_attention_shape():
    attn = SlidingWindowAttention(d_model=64, n_heads=8, dropout=0.0, window_size=16)
    x = torch.randn(2, 32, 64)
    y = attn(x)
    assert y.shape == x.shape


def test_ssm_forward_backward():
    mod = SimpleSelectiveSSM(SSMConfig(d_model=64, state_size=32, dropout=0.0))
    x = torch.randn(2, 24, 64, requires_grad=True)
    y = mod(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None


def test_hybrid_output_shape():
    cfg = HybridConfig(d_model=64, n_heads=8, n_local_layers=1, n_ssm_layers=1, max_seq_len=64)
    m = SplitStreamHybridLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = m(x, x)
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_param_count_nonzero():
    b = GPTBaseline(TransformerConfig(d_model=64, n_heads=8, n_layers=2, max_seq_len=64))
    assert count_params(b) > 0


def test_kan_linear_forward_backward():
    mod = KANLinear(KANConfig(in_features=16, out_features=8, n_basis=6))
    x = torch.randn(4, 16, requires_grad=True)
    y = mod(x)
    assert y.shape == (4, 8)
    y.mean().backward()
    assert x.grad is not None


def test_hybrid_kan_fusion_shape():
    cfg = HybridConfig(
        d_model=64,
        n_heads=8,
        n_local_layers=1,
        n_ssm_layers=1,
        max_seq_len=64,
        fusion_mode="kan",
        kan_basis=6,
    )
    m = SplitStreamHybridLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = m(x, x)
    assert logits.shape == (2, 32, cfg.vocab_size)
