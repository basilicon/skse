"""Simplex-volume losses and samplers used by SKSE."""
import torch

def simplex_volume(points, idx):
    """
    points: (N, d) tensor of y's (or x's)
    idx: (B, k+1) long tensor of indices defining B simplices
    returns: (B,) squared volumes Vol_k^2
    """
    B, k1 = idx.shape
    k = k1 - 1
    Y = points[idx]                   # (B, k+1, d)
    V = Y[:, 1:] - Y[:, :1]           # (B, k, d)
    
    # Gram matrix: (B, k, k)
    G = torch.matmul(V, V.transpose(1, 2))
    
    sign, logdet = torch.linalg.slogdet(G)  # sign ∈ {-1, 0, 1}, logdet ∈ R
    
    # safeguard logdet
    # logdet = torch.clamp(logdet, min=-25, max=25)

    # squared volume = exp(0.5 * logdet - log(k!))
    volumes = torch.exp(0.5 * logdet - torch.lgamma(torch.tensor(k)))
    
    # For degenerate simplices (sign <= 0), set volume to 0
    volumes = torch.where(sign <= 0, torch.tensor(0.0, device=volumes.device), volumes)
    
    return volumes

def universal_simplex_volume(points, idx):
    # volume of the simplex in its maximal non-degenerate affine subspace
    # this version uses SVD to compute volume, more robust to k > d or d > k
    # more expensive but fine for small k,d

    B, k1 = idx.shape
    k = k1 - 1
    d = points.shape[1]
    
    Y = points[idx]  # (B, k+1, d)
    V = Y[:, 1:] - Y[:, :1]  # (B, k, d)
    
    # SVD is robust to k > d or d > k
    # s shape: (B, min(k, d))
    s = torch.linalg.svdvals(V)
    
    # We want the volume of the 'actual' shape formed.
    # If d < k, we can only have at most d non-zero singular values.
    # If k < d, we have k singular values.
    # The effective dimension is m = min(k, d)
    m = s.size(-1)
    
    # Sum logs instead of product for stability
    # log(Vol) = sum(log(s)) - log(m!)
    log_vol = torch.sum(torch.log(s + 1e-12), dim=-1) - torch.lgamma(torch.tensor(m + 1.0))
    
    return torch.exp(log_vol)

def metric_skse_loss(x, y, simplices, normalize:None|str='batch', tdist=False):
    """
    x: (N, D) original points (no grads)
    y: (N, d) embedded points (requires_grad)
    simplices: (B, k+1) sampled tuples
    normalize: 'batch' | None | 'quality'
    lam_edges: optional edge stabilizer
    """

    vol2_x = simplex_volume(x, simplices).detach()  # target, no grad

    # if the embedded dim is less than k, use universal volume
    if y.shape[1] < simplices.shape[1] - 1:
        vol2_y = universal_simplex_volume(y, simplices)
    else:
        vol2_y = simplex_volume(y, simplices)


    if normalize == 'batch':
        vol2_x = vol2_x / (vol2_x.sum() + 1e-12)
        vol2_y = vol2_y / (vol2_y.sum() + 1e-12)
    elif normalize == 'quality':
        # divide by product of edge lengths^2 (scale invariance)
        def edge_prod2(pts):
            P = pts[simplices]  # (B,k+1,d)
            # all edges from i0 to others (k edges)
            E = P[:, 1:] - P[:, :1]  # (B,k,d)
            return (E.pow(2).sum(-1) + 1e-12).prod(-1)  # (B,)
        vol2_x = vol2_x / (edge_prod2(x).detach())
        vol2_y = vol2_y / (edge_prod2(y) + 1e-12)

    if tdist:
        # heavy-tailed variant
        p = vol2_x / (vol2_x.sum() + 1e-12)
        q_unnorm = (1.0 + vol2_y).pow(-1)
        q = q_unnorm / (q_unnorm.sum() + 1e-12)
        loss = (p * (p.add(1e-12).log() - q.add(1e-12).log())).sum()
    else:
        # standard MSE
        loss = (vol2_y - vol2_x).pow(2).mean()

    return loss


def sample_simplices(n, batch_size=1024, points=3):
    """
    Random simplex sampler:
    - choose batch_size permutations of [0, n) of size points
    """
    
    simplices = torch.stack([
        torch.randperm(n)[:points] for _ in range(batch_size)
    ], dim=0)
    return simplices
