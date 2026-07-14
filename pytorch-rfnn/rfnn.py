"""
PyTorch port of the Theano RFNN MNIST model from
"Structured Receptive Fields in CNNs" (Jacobsen et al., CVPR 2016).

Faithful re-implementation of theano-rfnn/run_rfnn.py + util.py.

Core idea: conv filters are NOT learned as free pixels. A fixed Gaussian-
derivative (Hermite) basis is precomputed; only the per-filter linear
combination coefficients ("alphas") are learned:

    weight[f, c] = sum_b  alpha[f, c, b] * basis[b]
"""

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_basis_hermite(sigma, bases, extent):
    """Port of util.init_basis_hermite. Returns (bases, imSize, imSize) float32."""
    x = np.arange(-extent, extent + 1, dtype=np.float64)
    imSize = extent * 2 + 1
    impulse = np.zeros((imSize, imSize))
    impulse[imSize // 2, imSize // 2] = 1.0
    nrBasis = 15
    hermiteBasis = np.empty((nrBasis, imSize, imSize))

    g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
    g = g / g.sum()
    g1 = sigma * -(x / np.square(sigma)) * g
    g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
    g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
    g4 = np.power(sigma, 4) * (((np.power(x, 4) - 6 * np.square(x) * np.square(sigma)
                                 + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g

    conv = ndi.convolve1d
    gauss0x = conv(impulse, g, axis=1)
    gauss0y = conv(impulse, g, axis=0)
    gauss1x = conv(impulse, g1, axis=1)
    gauss1y = conv(impulse, g1, axis=0)
    gauss2x = conv(impulse, g2, axis=1)
    gauss0 = conv(gauss0x, g, axis=0)

    hermiteBasis[0] = gauss0
    hermiteBasis[1] = conv(gauss0y, g1, axis=1)   # g_x
    hermiteBasis[2] = conv(gauss0x, g1, axis=0)   # g_y
    hermiteBasis[3] = conv(gauss0y, g2, axis=1)   # g_xx
    hermiteBasis[4] = conv(gauss0x, g2, axis=0)   # g_yy
    hermiteBasis[5] = conv(gauss1x, g1, axis=0)   # g_xy
    hermiteBasis[6] = conv(gauss0y, g3, axis=1)   # g_xxx
    hermiteBasis[7] = conv(gauss0x, g3, axis=0)   # g_yyy
    hermiteBasis[8] = conv(gauss1y, g2, axis=1)   # g_xxy
    hermiteBasis[9] = conv(gauss1x, g2, axis=0)   # g_yyx
    hermiteBasis[10] = conv(gauss0y, g4, axis=1)  # g_xxxx
    hermiteBasis[11] = conv(gauss0x, g4, axis=0)  # g_yyyy
    hermiteBasis[12] = conv(gauss1y, g3, axis=1)  # g_xxxy
    hermiteBasis[13] = conv(gauss1x, g3, axis=0)  # g_yyyx
    hermiteBasis[14] = conv(gauss2x, g2, axis=0)  # g_yyxx

    return torch.from_numpy(hermiteBasis[0:bases].astype(np.float32))


class BasisConv2d(nn.Module):
    """Convolution whose weights are a learned linear combo of a fixed basis."""

    def __init__(self, out_ch, in_ch, sigma, bases, extent, padding):
        super().__init__()
        basis = init_basis_hermite(sigma, bases, extent)      # (bases, k, k)
        self.register_buffer("basis", basis)                  # fixed, not trained
        # alphas ~ U(-1, 1), shape (out, in, bases)  -- these are the learned params
        self.alphas = nn.Parameter(torch.empty(out_ch, in_ch, bases).uniform_(-1.0, 1.0))
        self.padding = padding

    def forward(self, x):
        # weight[f,c,h,w] = sum_b alpha[f,c,b] * basis[b,h,w]
        weight = torch.einsum("fcb,bhw->fchw", self.alphas, self.basis)
        return F.conv2d(x, weight, padding=self.padding)


class RFNN(nn.Module):
    def __init__(self, p_drop_conv=0.2, p_drop_hidden=0.7):
        super().__init__()
        # L1: 64 filters, 1 in-channel, sigma=1.5, 10 bases, extent=5 -> 11x11, "same" pad 5
        self.conv1 = BasisConv2d(64, 1, sigma=1.5, bases=10, extent=5, padding=5)
        # L2/L3: sigma=1, 6 bases, extent=3 -> 7x7, "full" conv -> pad = k-1 = 6
        self.conv2 = BasisConv2d(64, 64, sigma=1.0, bases=6, extent=3, padding=6)
        self.conv3 = BasisConv2d(64, 64, sigma=1.0, bases=6, extent=3, padding=6)
        self.lrn = nn.LocalResponseNorm(size=9, alpha=1e-4, beta=0.75, k=2)
        self.drop_conv = nn.Dropout(p_drop_conv)
        self.drop_hidden = nn.Dropout(p_drop_hidden)
        # final linear 3136 -> 10 (64 * 7 * 7)
        self.fc = nn.Linear(3136, 10, bias=False)
        nn.init.normal_(self.fc.weight, std=0.01)

    def _block(self, x, conv, drop):
        x = F.relu(conv(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)   # ignore_border=True (floor)
        x = self.lrn(x)
        if drop:
            x = self.drop_conv(x)
        return x

    def forward(self, x):
        x = self._block(x, self.conv1, drop=True)
        x = self._block(x, self.conv2, drop=True)
        x = self._block(x, self.conv3, drop=False)
        x = torch.flatten(x, 1)
        x = self.drop_hidden(x)
        return self.fc(x)   # logits (softmax folded into CrossEntropyLoss)
