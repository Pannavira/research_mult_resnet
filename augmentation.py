"""
augmentation.py — Signal augmentation utilities for wearable sensor data.

Augmentations applied during training to improve model robustness against
inter-subject variability:
  - Gaussian Noise: adds random noise to improve noise robustness.
  - Random Scaling: multiplies the signal by a random factor (0.8–1.2)
    to handle amplitude variability across subjects.
  - Time Permutation: splits the signal into random segments and shuffles
    them to break temporal memorization while preserving local patterns.
"""

import torch
import numpy as np


def add_gaussian_noise(signal: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to a signal to improve robustness.
    
    Args:
        signal (torch.Tensor): The input signal, e.g. (2, seq_len)
        std (float): The standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: The noisy signal.
    """
    if std <= 0.0:
        return signal
    
    noise = torch.randn_like(signal) * std
    return signal + noise


def random_scaling(signal: torch.Tensor,
                   scale_min: float = 0.8,
                   scale_max: float = 1.2) -> torch.Tensor:
    """
    Multiply each channel by a random scaling factor to simulate
    inter-subject amplitude variability.

    A single scale factor is drawn per channel so the waveform shape
    is preserved while the overall amplitude changes.

    Args:
        signal (torch.Tensor): Input signal, shape ``(C, seq_len)``.
        scale_min (float): Lower bound of the uniform distribution.
        scale_max (float): Upper bound of the uniform distribution.

    Returns:
        torch.Tensor: Scaled signal, same shape as input.
    """
    # One random factor per channel: shape (C, 1)
    scale = torch.empty(signal.size(0), 1).uniform_(scale_min, scale_max)
    return signal * scale


def time_permutation(signal: torch.Tensor,
                     n_segments: int = 5) -> torch.Tensor:
    """
    Split the signal along the time axis into ``n_segments`` chunks and
    randomly shuffle their order.

    This forces the model to rely on local waveform patterns rather than
    memorizing the exact temporal position of events within a window.

    Args:
        signal (torch.Tensor): Input signal, shape ``(C, seq_len)``.
        n_segments (int): Number of equal-length chunks to create.

    Returns:
        torch.Tensor: Permuted signal, same shape as input.
    """
    seq_len = signal.size(1)
    seg_len = seq_len // n_segments

    # Only permute if segments are at least 1 sample long
    if seg_len < 1:
        return signal

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else seq_len
        segments.append(signal[:, start:end])

    # Shuffle segment order
    perm = torch.randperm(len(segments))
    segments = [segments[i] for i in perm]

    return torch.cat(segments, dim=1)


def augment_signal(signal: torch.Tensor,
                   noise_std: float = 0.05,
                   p_scaling: float = 0.5,
                   p_permutation: float = 0.3,
                   n_segments: int = 5) -> torch.Tensor:
    """
    Apply a stochastic augmentation pipeline to a multi-channel signal.

    Each augmentation is applied independently with its own probability,
    so a single sample may receive zero, one, or multiple augmentations.

    Args:
        signal (torch.Tensor): Input signal, shape ``(C, seq_len)``.
        noise_std (float): Std-dev for Gaussian noise (always applied if > 0).
        p_scaling (float): Probability of applying random scaling.
        p_permutation (float): Probability of applying time permutation.
        n_segments (int): Number of segments for time permutation.

    Returns:
        torch.Tensor: Augmented signal, same shape as input.
    """
    # Gaussian noise — always applied (lightweight)
    signal = add_gaussian_noise(signal, std=noise_std)

    # Random scaling — 50 % chance
    if torch.rand(1).item() < p_scaling:
        signal = random_scaling(signal)

    # Time permutation — 30 % chance
    if torch.rand(1).item() < p_permutation:
        signal = time_permutation(signal, n_segments=n_segments)

    return signal
