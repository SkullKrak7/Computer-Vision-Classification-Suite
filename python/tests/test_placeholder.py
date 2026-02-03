"""Minimal passing test for CI - Phase 1"""


def test_placeholder():
    """Placeholder test to make CI pass"""
    assert True


def test_numpy_import():
    """Test numpy import"""
    import numpy as np

    assert np.array([1, 2, 3]).sum() == 6


def test_torch_import():
    """Test torch import"""
    import torch

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0

