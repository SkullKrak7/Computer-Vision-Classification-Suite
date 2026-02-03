"""Tests for image validation utilities"""

import pytest
from fastapi import HTTPException

from backend.app.utils.validation import sanitize_filename


def test_sanitize_filename_path_traversal():
    """Test path traversal prevention"""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("../../../secret.txt") == "secret.txt"


def test_sanitize_filename_special_chars():
    """Test special character removal"""
    result = sanitize_filename("file<>name.jpg")
    assert "<" not in result and ">" not in result
    result = sanitize_filename("test|file?.png")
    assert "|" not in result and "?" not in result
