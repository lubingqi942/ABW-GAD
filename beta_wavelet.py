import torch
import torch.nn as nn
import numpy as np
from scipy.special import beta as beta_func


_HARDCODED_THETAS = {
    1: [
        [2.0, -1.0],
        [0.0, 1.0],
    ],
    2: [
        [3.0, -3.0, 0.75],
        [0.0, 3.0, -1.5],
        [0.0, 0.0, 0.75],
    ],
    3: [
        [4.0, -6.0, 3.0, -0.5],
        [0.0, 6.0, -6.0, 1.5],
        [0.0, 0.0, 3.0, -1.5],
        [0.0, 0.0, 0.0, 0.5],
    ],
    4: [
        [5.0, -10.0, 7.5, -2.5, 0.3125],
        [0.0, 10.0, -15.0, 7.5, -1.25],
        [0.0, 0.0, 7.5, -7.5, 1.875],
        [0.0, 0.0, 0.0, 2.5, -1.25],
        [0.0, 0.0, 0.0, 0.0, 0.3125],
    ],
    5: [
        [6.0, -15.0, 15.0, -7.5, 1.875, -0.1875],
        [0.0, 15.0, -30.0, 22.5, -7.5, 0.9375],
        [0.0, 0.0, 15.0, -22.5, 11.25, -1.875],
        [0.0, 0.0, 0.0, 7.5, -7.5, 1.875],
        [0.0, 0.0, 0.0, 0.0, 1.875, -0.9375],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1875],
    ],
    6: [
        [7.0, -21.0, 26.25, -17.5, 6.5625, -1.3125, 0.109375],
        [0.0, 21.0, -52.5, 52.5, -26.25, 6.5625, -0.65625],
        [0.0, 0.0, 26.25, -52.5, 39.375, -13.125, 1.640625],
        [0.0, 0.0, 0.0, 17.5, -26.25, 13.125, -2.1875],
        [0.0, 0.0, 0.0, 0.0, 6.5625, -6.5625, 1.640625],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.3125, -0.65625],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.109375],
    ],
}


def calculate_beta_wavelet_theta(d):
    return calculate_beta_wavelet_theta_general(d, beta_shape=1.0)


def calculate_beta_wavelet_theta_general(d, beta_shape=1.0):
    if abs(beta_shape - 1.0) < 1e-10 and d in _HARDCODED_THETAS:
        return _HARDCODED_THETAS[d]

    try:
        import sympy
    except ImportError:
        raise ImportError(
            f"Beta wavelet with beta_shape={beta_shape} requires sympy "
            f"(only beta_shape=1.0 is hardcoded). Install: pip install sympy"
        )

    x = sympy.symbols('x')
    thetas = []
    s = beta_shape

    for i in range(d + 1):
        alpha = i * s + 1
        beta_param = (d - i) * s + 1
        expr = (x / 2) ** i * (1 - x / 2) ** (d - i) / beta_func(alpha, beta_param)
        poly = sympy.poly(expr, x)
        coeffs = poly.all_coeffs()
        inv_coeffs = [float(coeffs[d - k]) for k in range(d + 1)]
        thetas.append(inv_coeffs)

    return thetas


_THETA_CACHE = {}

def get_beta_theta(d, beta_shape=1.0):
    key = (d, beta_shape)
    if key not in _THETA_CACHE:
        _THETA_CACHE[key] = calculate_beta_wavelet_theta_general(d, beta_shape)
    return _THETA_CACHE[key]


class BetaWaveletFilter(nn.Module):

    def __init__(self, theta):
        super(BetaWaveletFilter, self).__init__()
        self.register_buffer('theta', torch.tensor(theta, dtype=torch.float32))
        self.K = len(theta)

    def forward(self, x, adj_norm):
        h = self.theta[0] * x
        Lkx = x

        for k in range(1, self.K):
            Lkx = Lkx - torch.sparse.mm(adj_norm, Lkx)
            h = h + self.theta[k] * Lkx

        return h


class MultiScaleBetaWavelet(nn.Module):

    def __init__(self, d=2, beta_shape=1.0):
        super(MultiScaleBetaWavelet, self).__init__()
        self.d = d
        self.beta_shape = beta_shape
        self.num_filters = d + 1

        all_thetas = get_beta_theta(d, beta_shape)

        self.filters = nn.ModuleList([
            BetaWaveletFilter(theta) for theta in all_thetas
        ])

    def forward(self, x, adj_norm):
        return [filt(x, adj_norm) for filt in self.filters]


if __name__ == '__main__':
    print("=== Beta Wavelet Coefficient Test ===")
    for d in [1, 2, 3, 4]:
        thetas = get_beta_theta(d)
        print(f"\nOrder d={d} ({len(thetas)} filters):")
        for i, th in enumerate(thetas):
            print(f"  Filter {i}: theta = {[f'{v:.4f}' for v in th]}")

    print("\n=== MultiScaleBetaWavelet Test ===")
    N, F = 50, 10
    x = torch.randn(N, F)
    adj_dense = torch.zeros(N, N)
    for i in range(N):
        adj_dense[i, (i - 1) % N] = 1.0
        adj_dense[i, (i + 1) % N] = 1.0
    deg = adj_dense.sum(dim=1).pow(-0.5)
    adj_norm = deg.unsqueeze(1) * adj_dense * deg.unsqueeze(0)
    adj_norm = adj_norm.to_sparse()

    for d in [1, 2, 3]:
        wavelet = MultiScaleBetaWavelet(d=d)
        outputs = wavelet(x, adj_norm)
        print(f"\nd={d}: {wavelet.num_filters} filters (one-to-one bands)")
        for idx, h in enumerate(outputs):
            print(f"  Band {idx}: {list(h.shape)}")
