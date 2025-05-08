"""
Voice recognition module for OpenSCAD voice control.
This module provides functionality for converting speech to text.
"""

from .recognizer import VoiceRecognizer
from .microphone_manager import MicrophoneManager

__all__ = ['VoiceRecognizer', 'MicrophoneManager']
