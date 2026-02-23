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
