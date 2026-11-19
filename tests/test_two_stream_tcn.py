"""Stage 3.6.7 — two-stream TCN shapes and multi-task loss."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from ecgdet.two_stream_tcn import TwoStreamTCN, multitask_loss


def test_forward_shapes_and_finite_loss():
    torch.manual_seed(0)
    net = TwoStreamTCN(in_channels=12, length=5000)
    x = torch.randn(2, 12, 5000)
    q, a, logit = net(x)
    assert q.shape == x.shape and a.shape == x.shape
    assert logit.shape == (2,)
    y = torch.tensor([1.0, 0.0])
    loss, parts = multitask_loss(x, q, a, logit, y)
    assert torch.isfinite(loss)
    assert parts["atr"] is None


def test_atrial_supervision_is_differentiable():
    torch.manual_seed(1)
    net = TwoStreamTCN(in_channels=12, length=5000)
    x = torch.randn(2, 12, 5000)
    q, a, logit = net(x)
    y = torch.tensor([1.0, 0.0])
    loss, parts = multitask_loss(x, q, a, logit, y, atrial_true=torch.zeros_like(a))
    loss.backward()
    assert parts["atr"] is not None
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_short_synthetic_training_reduces_atrial_error():
    torch.manual_seed(2)
    net = TwoStreamTCN(in_channels=1, length=1000, base=16)
    n = 1000
    t = torch.linspace(0, 1, n)
    atrial_true = (0.2 * torch.sin(2 * torch.pi * 5 * t)).view(1, 1, n).repeat(4, 1, 1)
    qrs = torch.zeros_like(atrial_true)
    qrs[:, :, 400:420] = 1.0
    x = qrs + atrial_true
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    def err():
        with torch.no_grad():
            _, a, _ = net(x)
            return float(((a - atrial_true) ** 2).mean())
    before = err()
    for _ in range(15):
        q, a, logit = net(x)
        loss, _ = multitask_loss(x, q, a, logit, y, atrial_true=atrial_true,
                                 w_rec=1.0, w_atr=5.0, w_cls=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    assert err() < before, (before, err())
