import torch

from lanobert.fusion import GatedFusion, freeze_baseline


def test_gated_fusion_returns_batch_scores_and_normalized_gate():
    fusion = GatedFusion(8, 4, 4, hidden_dim=6)
    scores = fusion(torch.randn(3, 8), torch.randn(3, 4), torch.ones(3, 4))
    assert scores.shape == (3,)
    assert torch.allclose(fusion.last_gate.sum(dim=-1), torch.ones(3))


def test_freeze_baseline_disables_gradients_and_training_mode():
    baseline = torch.nn.Linear(4, 2)
    assert freeze_baseline(baseline).training is False
    assert all(not parameter.requires_grad for parameter in baseline.parameters())
