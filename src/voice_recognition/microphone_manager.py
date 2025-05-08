"""
Microphone manager for handling audio input.
"""

import speech_recognition as sr
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MicrophoneManager:
    """
    Manages microphone input and provides methods for audio capture.
    """
    
    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        """
        Initialize the microphone manager.
        
        Args:
            device_index (int, optional): Index of the microphone device to use.
                                         None uses the default microphone.
            sample_rate (int): Sample rate for audio recording.
            chunk_size (int): Size of audio chunks to process.
        """
        self.recognizer = sr.Recognizer()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.microphone = None
        
        # Configure recognizer settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300  # Default energy threshold
        self.recognizer.pause_threshold = 0.8   # Seconds of silence before considering the phrase complete
        
        logger.info("Microphone manager initialized")
    
    def list_microphone_devices(self):
        """
        List all available microphone devices.
        
        Returns:
            list: Available microphone devices with their indices.
        """
        devices = []
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            devices.append({"index": index, "name": name})
        return devices
    
    def start_listening(self):
        """
        Start listening to the microphone.
        
        Returns:
            sr.Microphone: The active microphone source.
        """
        try:
            self.microphone = sr.Microphone(
                device_index=self.device_index,
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size
            )
            logger.info(f"Started listening with device index: {self.device_index}")
            return self.microphone
        except Exception as e:
            logger.error(f"Error starting microphone: {e}")
            raise
    
    def adjust_for_ambient_noise(self, duration=1):
        """
        Adjust recognizer for ambient noise.
        
        Args:
            duration (float): Duration in seconds to sample ambient noise.
        """
        if not self.microphone:
            self.start_listening()
            
        try:
            with self.microphone as source:
                logger.info(f"Adjusting for ambient noise (duration: {duration}s)")
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
                logger.info(f"Energy threshold adjusted to: {self.recognizer.energy_threshold}")
        except Exception as e:
            logger.error(f"Error adjusting for ambient noise: {e}")
            raise
