"""Pytest fixtures for Python ML tests"""

import numpy as np
import pytest


@pytest.fixture
def sample_image_array():
    """Generate sample image array"""
    return np.random.rand(64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Generate sample batch"""
    return np.random.rand(4, 64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels"""
    return np.array([0, 1, 2, 3])
