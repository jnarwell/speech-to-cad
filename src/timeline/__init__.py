"""
Timeline module for tracking operations history.
This module provides functionality for tracking the history of CAD operations.
"""

from .operation import Operation
from .timeline_manager import TimelineManager

__all__ = ['Operation', 'TimelineManager']
