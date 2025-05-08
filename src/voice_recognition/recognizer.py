"""
Voice recognition module for converting speech to text.
"""

import speech_recognition as sr
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceRecognizer:
    """
    Handles voice recognition and converts speech to text.
    """
    
    def __init__(self, api_key=None, use_google=False, use_whisper=False):
        """
        Initialize the voice recognizer.
        
        Args:
            api_key (str, optional): API key for speech recognition service.
                                   If None, will try to get from environment variables.
            use_google (bool): Whether to use Google Speech Recognition.
            use_whisper (bool): Whether to use OpenAI's Whisper API.
        """
        self.recognizer = sr.Recognizer()
        
        # Set API key
        if api_key is None:
            if use_whisper:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("OpenAI API key not found in environment variables")
            elif use_google:
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    logger.warning("Google API key not found in environment variables")
        else:
            self.api_key = api_key
            
        self.use_google = use_google
        self.use_whisper = use_whisper
        
        logger.info("Voice recognizer initialized")
        
    def recognize_from_microphone(self, microphone_source, timeout=5):
        """
        Recognize speech from a microphone source.
        
        Args:
            microphone_source: Microphone source from MicrophoneManager.
            timeout (int): Maximum duration in seconds to listen for a phrase.
            
        Returns:
            str: Recognized text.
            
        Raises:
            sr.UnknownValueError: If speech cannot be understood.
            sr.RequestError: If the recognition service fails.
        """
        try:
            with microphone_source as source:
                logger.info("Listening for input...")
                audio = self.recognizer.listen(source, timeout=timeout)
                logger.info("Audio captured, processing...")
                
                return self.recognize_audio(audio)
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return None
        except Exception as e:
            logger.error(f"Error during voice recognition: {e}")
            raise
            
    def recognize_audio(self, audio):
        """
        Recognize speech from an audio source.
        
        Args:
            audio: Audio data to recognize.
            
        Returns:
            str: Recognized text.
            
        Raises:
            sr.UnknownValueError: If speech cannot be understood.
            sr.RequestError: If the recognition service fails.
        """
        try:
            if self.use_whisper and self.api_key:
                # Use OpenAI's Whisper API
                text = self.recognizer.recognize_whisper_api(
                    audio, 
                    api_key=self.api_key
                )
                logger.info("Speech recognized using Whisper API")
                return text
            elif self.use_google and self.api_key:
                # Use Google Cloud Speech Recognition
                text = self.recognizer.recognize_google_cloud(
                    audio, 
                    credentials_json=self.api_key
                )
                logger.info("Speech recognized using Google Cloud")
                return text
            else:
                # Use Google's free API (limited usage)
                text = self.recognizer.recognize_google(audio)
                logger.info("Speech recognized using Google free API")
                return text
                
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during recognition: {e}")
            return None
