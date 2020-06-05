import os
import torch
import inspect
import pytest

current_file_dir = os.path.dirname(__file__)
platform = 'cuda' if torch.cuda.is_available() else 'cpu'
# Get information about test being skipped
frame = inspect.currentframe()
frame_info = inspect.getframeinfo(frame)

# Decorator function to skip a test unless torch.cuda.is_available() is true
gpu_is_available = pytest.mark.skipif(
    platform != 'cuda',
    reason=f"On platform {platform.upper()}, the test at line {frame_info.lineno} in {frame_info.filename} is not executed."
)
