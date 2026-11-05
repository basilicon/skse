import numpy as np
import torch

from skse import SKSE, metric_skse_loss, sample_simplices, simplex_volume


def test_sample_simplices_shape_and_distinct():
    s = sample_simplices(10, batch_size=7, points=4)
    assert s.shape == (7, 4)
    for row in s:
        assert len(set(row.tolist())) == 4


def test_simplex_volume_scales_with_size():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    idx = torch.tensor([[0, 1, 2]])
    v1 = simplex_volume(pts, idx)
    v2 = simplex_volume(pts * 2, idx)
    # area scales by 2^2 when lengths double
    assert torch.allclose(v2, v1 * 4)


def test_loss_is_zero_for_identical_points():
    x = torch.randn(20, 5)
    idx = sample_simplices(20, batch_size=8, points=3)
    assert metric_skse_loss(x, x.clone(), idx).item() < 1e-12


def test_fit_transform_shape():
    X = np.random.RandomState(0).randn(40, 10).astype(np.float32)
    Y = SKSE(n_components=3, n_points=3, epochs=2, random_state=0, device="cpu").fit_transform(X)
    assert Y.shape == (40, 3)
    assert np.isfinite(Y).all()


def test_random_state_is_deterministic():
    X = np.random.RandomState(0).randn(30, 6).astype(np.float32)
    kw = dict(n_components=2, n_points=3, epochs=3, random_state=123, device="cpu")
    np.testing.assert_allclose(SKSE(**kw).fit_transform(X), SKSE(**kw).fit_transform(X))
