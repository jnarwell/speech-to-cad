#!/bin/bash
# Create the directory structure
mkdir -p src/voice_recognition
mkdir -p src/openscad_interface
mkdir -p src/timeline
mkdir -p src/nlp_processor
mkdir -p database
mkdir -p models
mkdir -p tests
mkdir -p docs

# Create __init__.py files
touch src/__init__.py
touch src/voice_recognition/__init__.py
touch src/openscad_interface/__init__.py
touch src/timeline/__init__.py
touch src/nlp_processor/__init__.py

# Create voice_recognition module files
cat > src/voice_recognition/__init__.py << 'EOF'
"""
Voice recognition module for OpenSCAD voice control.
This module provides functionality for converting speech to text.
"""

from .recognizer import VoiceRecognizer
from .microphone_manager import MicrophoneManager

__all__ = ['VoiceRecognizer', 'MicrophoneManager']
EOF

cat > src/voice_recognition/microphone_manager.py << 'EOF'
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
EOF

cat > src/voice_recognition/recognizer.py << 'EOF'
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
EOF

# Create openscad_interface module files
cat > src/openscad_interface/__init__.py << 'EOF'
"""
OpenSCAD interface module for generating and executing OpenSCAD code.
"""

from .code_generator import OpenSCADCodeGenerator
from .renderer import OpenSCADRenderer

__all__ = ['OpenSCADCodeGenerator', 'OpenSCADRenderer']
EOF

cat > src/openscad_interface/code_generator.py << 'EOF'
"""
OpenSCAD code generator module.
This module handles the generation of OpenSCAD code from processed commands.
"""

import logging
from solid import *
from solid.utils import *
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenSCADCodeGenerator:
    """
    Generates OpenSCAD code from processed commands.
    Uses SolidPython as a wrapper around OpenSCAD.
    """
    
    def __init__(self):
        """Initialize the OpenSCAD code generator."""
        self.current_model = None
        self.shapes = {}  # Dictionary to store named shapes
        self.current_transformation = None
        logger.info("OpenSCAD code generator initialized")
    
    def reset(self):
        """Reset the current model and shapes."""
        self.current_model = None
        self.shapes = {}
        self.current_transformation = None
        logger.info("OpenSCAD code generator reset")
    
    def generate_code(self):
        """
        Generate OpenSCAD code from the current model.
        
        Returns:
            str: The generated OpenSCAD code.
        """
        if self.current_model is None:
            logger.warning("No model to generate code from")
            return ""
        
        try:
            # Generate the OpenSCAD code using SolidPython
            code = scad_render(self.current_model)
            logger.info("OpenSCAD code generated successfully")
            return code
        except Exception as e:
            logger.error(f"Error generating OpenSCAD code: {e}")
            return ""
    
    def save_code(self, filename):
        """
        Save the generated OpenSCAD code to a file.
        
        Args:
            filename (str): The filename to save the code to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.current_model is None:
            logger.warning("No model to save")
            return False
        
        try:
            # Generate and save the OpenSCAD code
            scad_render_to_file(self.current_model, filename)
            logger.info(f"OpenSCAD code saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving OpenSCAD code: {e}")
            return False
    
    # === 2D Shape Operations ===
    
    def create_circle(self, radius, name=None):
        """
        Create a circle with the given radius.
        
        Args:
            radius (float): The radius of the circle.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The created circle object.
        """
        try:
            shape = circle(r=radius)
            
            # Store the shape if a name is provided
            if name:
                self.shapes[name] = shape
                logger.info(f"Created circle '{name}' with radius {radius}")
            else:
                logger.info(f"Created circle with radius {radius}")
            
            # Update current model if none exists
            if self.current_model is None:
                self.current_model = shape
            
            return shape
        except Exception as e:
            logger.error(f"Error creating circle: {e}")
            return None
    
    def create_square(self, size, center=True, name=None):
        """
        Create a square with the given size.
        
        Args:
            size (float or list): The size of the square. If a float, creates a
                                square with equal sides. If a list [x, y], creates
                                a rectangle.
            center (bool): Whether to center the square at the origin.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The created square object.
        """
        try:
            shape = square(size=size, center=center)
            
            # Store the shape if a name is provided
            if name:
                self.shapes[name] = shape
                if isinstance(size, (list, tuple)):
                    logger.info(f"Created rectangle '{name}' with size {size}")
                else:
                    logger.info(f"Created square '{name}' with size {size}")
            else:
                if isinstance(size, (list, tuple)):
                    logger.info(f"Created rectangle with size {size}")
                else:
                    logger.info(f"Created square with size {size}")
            
            # Update current model if none exists
            if self.current_model is None:
                self.current_model = shape
            
            return shape
        except Exception as e:
            logger.error(f"Error creating square: {e}")
            return None
    
    def create_polygon(self, points, name=None):
        """
        Create a polygon with the given points.
        
        Args:
            points (list): List of points defining the polygon.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The created polygon object.
        """
        try:
            shape = polygon(points=points)
            
            # Store the shape if a name is provided
            if name:
                self.shapes[name] = shape
                logger.info(f"Created polygon '{name}' with {len(points)} points")
            else:
                logger.info(f"Created polygon with {len(points)} points")
            
            # Update current model if none exists
            if self.current_model is None:
                self.current_model = shape
            
            return shape
        except Exception as e:
            logger.error(f"Error creating polygon: {e}")
            return None
    
    # === 3D Operations ===
    
    def extrude_linear(self, shape, height, center=False, name=None):
        """
        Linearly extrude a 2D shape into 3D.
        
        Args:
            shape: The 2D shape to extrude. Can be a name or an object.
            height (float): The height to extrude.
            center (bool): Whether to center the extrusion along the Z axis.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The extruded 3D object.
        """
        try:
            # If shape is a string, look it up in the shapes dictionary
            if isinstance(shape, str):
                if shape in self.shapes:
                    shape_obj = self.shapes[shape]
                else:
                    logger.error(f"Shape with name '{shape}' not found")
                    return None
            else:
                shape_obj = shape
            
            # Perform the extrusion
            extruded = linear_extrude(height=height, center=center)(shape_obj)
            
            # Store the extruded shape if a name is provided
            if name:
                self.shapes[name] = extruded
                logger.info(f"Created linear extrusion '{name}' with height {height}")
            else:
                logger.info(f"Created linear extrusion with height {height}")
            
            # Update the current model
            self.current_model = extruded
            
            return extruded
        except Exception as e:
            logger.error(f"Error creating linear extrusion: {e}")
            return None
    
    def extrude_rotate(self, shape, angle=360, name=None):
        """
        Create a rotational extrusion of a 2D shape.
        
        Args:
            shape: The 2D shape to extrude. Can be a name or an object.
            angle (float): The angle to rotate the extrusion.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The rotated 3D object.
        """
        try:
            # If shape is a string, look it up in the shapes dictionary
            if isinstance(shape, str):
                if shape in self.shapes:
                    shape_obj = self.shapes[shape]
                else:
                    logger.error(f"Shape with name '{shape}' not found")
                    return None
            else:
                shape_obj = shape
            
            # Perform the rotation extrusion
            extruded = rotate_extrude(angle=angle)(shape_obj)
            
            # Store the extruded shape if a name is provided
            if name:
                self.shapes[name] = extruded
                logger.info(f"Created rotate extrusion '{name}' with angle {angle}")
            else:
                logger.info(f"Created rotate extrusion with angle {angle}")
            
            # Update the current model
            self.current_model = extruded
            
            return extruded
        except Exception as e:
            logger.error(f"Error creating rotation extrusion: {e}")
            return None
    
    # === Transformation Operations ===
    
    def mirror(self, shape, vector, name=None):
        """
        Mirror a shape across a plane defined by a vector.
        
        Args:
            shape: The shape to mirror. Can be a name or an object.
            vector (list): The vector defining the mirror plane.
            name (str, optional): Name to assign to this shape.
            
        Returns:
            object: The mirrored object.
        """
        try:
            # If shape is a string, look it up in the shapes dictionary
            if isinstance(shape, str):
                if shape in self.shapes:
                    shape_obj = self.shapes[shape]
                else:
                    logger.error(f"Shape with name '{shape}' not found")
                    return None
            else:
                shape_obj = shape
            
            # Perform the mirror operation
            mirrored = mirror(vector)(shape_obj)
            
            # Store the mirrored shape if a name is provided
            if name:
                self.shapes[name] = mirrored
                logger.info(f"Created mirror '{name}' with vector {vector}")
            else:
                logger.info(f"Created mirror with vector {vector}")
            
            # Update the current model
            self.current_model = mirrored
            
            return mirrored
        except Exception as e:
            logger.error(f"Error creating mirror: {e}")
            return None
EOF

cat > src/openscad_interface/renderer.py << 'EOF'
"""
OpenSCAD renderer module.
This module handles rendering OpenSCAD code to create 3D models.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenSCADRenderer:
    """
    Renders OpenSCAD code to create 3D models.
    Handles interaction with the OpenSCAD program.
    """
    
    def __init__(self, openscad_path=None):
        """
        Initialize the OpenSCAD renderer.
        
        Args:
            openscad_path (str, optional): Path to the OpenSCAD executable.
                                          If None, will look for it in standard locations
                                          or use from environment variables.
        """
        # Get OpenSCAD path from environment variable if not provided
        if openscad_path is None:
            openscad_path = os.getenv("OPENSCAD_PATH")
            
        # Default paths for different operating systems
        if openscad_path is None:
            if os.name == 'nt':  # Windows
                possible_paths = [
                    r"C:\Program Files\OpenSCAD\openscad.exe",
                    r"C:\Program Files (x86)\OpenSCAD\openscad.exe"
                ]
            elif os.name == 'posix':  # Linux/Mac
                possible_paths = [
                    "/usr/bin/openscad",
                    "/usr/local/bin/openscad",
                    "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
                ]
            else:
                possible_paths = []
                
            # Find the first path that exists
            for path in possible_paths:
                if os.path.exists(path):
                    openscad_path = path
                    break
        
        self.openscad_path = openscad_path
        
        if self.openscad_path is None or not os.path.exists(self.openscad_path):
            logger.warning("OpenSCAD executable not found. Rendering will not be available.")
        else:
            logger.info(f"OpenSCAD renderer initialized with path: {self.openscad_path}")
            
        # Create models directory if it doesn't exist
        self.models_dir = Path("../models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def render(self, scad_code, output_file=None, format="stl"):
        """
        Render OpenSCAD code to create a 3D model file.
        
        Args:
            scad_code (str): The OpenSCAD code to render.
            output_file (str, optional): The output filename.
                                       If None, a temporary file will be used.
            format (str): The output format (stl, off, amf, 3mf, etc.)
            
        Returns:
            str: The path to the rendered file.
        """
        if self.openscad_path is None or not os.path.exists(self.openscad_path):
            logger.error("OpenSCAD executable not found. Cannot render.")
            return None
        
        try:
            # Create a temporary SCAD file
            with tempfile.NamedTemporaryFile(suffix='.scad', delete=False) as tmp_scad:
                tmp_scad.write(scad_code.encode('utf-8'))
                tmp_scad_path = tmp_scad.name
            
            # Determine output file path
            if output_file is None:
                output_file = os.path.join(
                    self.models_dir, 
                    f"model_{int(time.time())}.{format}"
                )
            
            # Build command to run OpenSCAD
            cmd = [
                self.openscad_path,
                "-o", output_file,
                tmp_scad_path
            ]
            
            # Run OpenSCAD to render the model
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Clean up temporary file
            os.unlink(tmp_scad_path)
            
            if process.returncode != 0:
                logger.error(f"OpenSCAD rendering failed: {process.stderr}")
                return None
            
            logger.info(f"OpenSCAD model rendered to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error rendering OpenSCAD model: {e}")
            return None
    
    def preview(self, scad_code):
        """
        Open OpenSCAD with the given code for preview.
        
        Args:
            scad_code (str): The OpenSCAD code to preview.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.openscad_path is None or not os.path.exists(self.openscad_path):
            logger.error("OpenSCAD executable not found. Cannot preview.")
            return False
        
        try:
            # Create a temporary SCAD file
            with tempfile.NamedTemporaryFile(suffix='.scad', delete=False) as tmp_scad:
                tmp_scad.write(scad_code.encode('utf-8'))
                tmp_scad_path = tmp_scad.name
            
            # Build command to run OpenSCAD in preview mode
            cmd = [
                self.openscad_path,
                tmp_scad_path
            ]
            
            # Run OpenSCAD for preview (non-blocking)
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"OpenSCAD preview opened")
            return True
            
        except Exception as e:
            logger.error(f"Error opening OpenSCAD preview: {e}")
            return False
EOF

# Create timeline module files
cat > src/timeline/__init__.py << 'EOF'
"""
Timeline module for tracking operations history.
This module provides functionality for tracking the history of CAD operations.
"""

from .operation import Operation
from .timeline_manager import TimelineManager

__all__ = ['Operation', 'TimelineManager']
EOF

cat > src/timeline/operation.py << 'EOF'
"""
Timeline operation module.
This module defines the Operation class for tracking CAD operations.
"""

import time
import uuid
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Operation:
    """
    Represents a single CAD operation in the timeline.
    """
    
    def __init__(self, operation_type, parameters=None, description=None):
        """
        Initialize a new operation.
        
        Args:
            operation_type (str): Type of operation (e.g., 'sketch', 'extrude', 'fillet').
            parameters (dict, optional): Parameters used for the operation.
            description (str, optional): Human-readable description of the operation.
        """
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.operation_type = operation_type
        self.parameters = parameters or {}
        self.description = description or f"{operation_type.capitalize()} operation"
        
        logger.info(f"Created operation: {self.description}")
    
    @property
    def readable_timestamp(self):
        """
        Get a human-readable timestamp.
        
        Returns:
            str: Human-readable timestamp.
        """
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))
    
    def to_dict(self):
        """
        Convert the operation to a dictionary.
        
        Returns:
            dict: Dictionary representation of the operation.
        """
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'readable_timestamp': self.readable_timestamp,
            'operation_type': self.operation_type,
            'parameters': self.parameters,
            'description': self.description
        }
    
    def to_json(self):
        """
        Convert the operation to a JSON string.
        
        Returns:
            str: JSON representation of the operation.
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data):
        """
        Create an operation from a dictionary.
        
        Args:
            data (dict): Dictionary representation of the operation.
            
        Returns:
            Operation: The created operation.
        """
        operation = cls(
            operation_type=data['operation_type'],
            parameters=data.get('parameters', {}),
            description=data.get('description', '')
        )
        operation.id = data['id']
        operation.timestamp = data['timestamp']
        
        return operation
    
    @classmethod
    def from_json(cls, json_str):
        """
        Create an operation from a JSON string.
        
        Args:
            json_str (str): JSON representation of the operation.
            
        Returns:
            Operation: The created operation.
        """
        return cls.from_dict(json.loads(json_str))
    
    def __str__(self):
        """
        Get a string representation of the operation.
        
        Returns:
            str: String representation of the operation.
        """
        return f"{self.readable_timestamp}: {self.description}"
    
    def __repr__(self):
        """
        Get a representation of the operation.
        
        Returns:
            str: String representation of the operation.
        """
        return f"Operation({self.operation_type}, {self.parameters}, {self.description})"
EOF

cat > src/timeline/timeline_manager.py << 'EOF'
"""
Timeline manager module.
This module handles the management of operation timelines.
"""

import os
import json
import logging
from pathlib import Path
import sqlite3
from datetime import datetime

from .operation import Operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimelineManager:
    """
    Manages timelines for CAD operations.
    """
    
    def __init__(self, database_path=None):
        """
        Initialize the timeline manager.
        
        Args:
            database_path (str, optional): Path to the database file.
                                         If None, a default path will be used.
        """
        # Set up database path
        if database_path is None:
            self.database_dir = Path("../database")
            self.database_dir.mkdir(parents=True, exist_ok=True)
            self.database_path = self.database_dir / "timeline.db"
        else:
            self.database_path = Path(database_path)
            self.database_dir = self.database_path.parent
            self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize timeline
        self.current_timeline = []
        self.current_position = -1
        
        logger.info(f"Timeline manager initialized with database: {self.database_path}")
    
    def _init_database(self):
        """Initialize the SQLite database for storing timelines."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create operations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    operation_type TEXT,
                    parameters TEXT,
                    description TEXT,
                    session_id TEXT
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at REAL,
                    last_modified REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def create_session(self, name=None):
        """
        Create a new session.
        
        Args:
            name (str, optional): Name for the session.
                                If None, a timestamp-based name will be used.
            
        Returns:
            str: ID of the created session.
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()
        
        if name is None:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO sessions (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                (session_id, name, timestamp, timestamp)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created session: {name} (ID: {session_id})")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def add_operation(self, operation, session_id):
        """
        Add an operation to the timeline.
        
        Args:
            operation (Operation): The operation to add.
            session_id (str): The session ID to add the operation to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Add to database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO operations (id, timestamp, operation_type, parameters, description, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation.id, 
                    operation.timestamp, 
                    operation.operation_type, 
                    json.dumps(operation.parameters), 
                    operation.description,
                    session_id
                )
            )
            
            # Update session last_modified timestamp
            cursor.execute(
                "UPDATE sessions SET last_modified = ? WHERE id = ?",
                (datetime.now().timestamp(), session_id)
            )
            
            conn.commit()
            conn.close()
            
            # Add to current timeline
            self.current_timeline.append(operation)
            self.current_position = len(self.current_timeline) - 1
            
            logger.info(f"Added operation to timeline: {operation.description}")
            return True
        except Exception as e:
            logger.error(f"Error adding operation to timeline: {e}")
            return False
    
    def get_operations(self, session_id, limit=None):
        """
        Get operations for a session.
        
        Args:
            session_id (str): The session ID to get operations for.
            limit (int, optional): Maximum number of operations to retrieve.
            
        Returns:
            list: List of operations.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if limit:
                cursor.execute(
                    "SELECT * FROM operations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM operations WHERE session_id = ? ORDER BY timestamp",
                    (session_id,)
                )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Retrieved {len(operations)} operations for session {session_id}")
            return operations
        except Exception as e:
            logger.error(f"Error retrieving operations: {e}")
            return []
    
    def get_sessions(self):
        """
        Get all sessions.
        
        Returns:
            list: List of sessions.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[2],
                    'last_modified': row[3]
                })
            
            conn.close()
            
            logger.info(f"Retrieved {len(sessions)} sessions")
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []
    
    def load_session(self, session_id):
        """
        Load a session into the current timeline.
        
        Args:
            session_id (str): The session ID to load.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            operations = self.get_operations(session_id)
            
            self.current_timeline = operations
            self.current_position = len(operations) - 1 if operations else -1
            
            logger.info(f"Loaded session {session_id} with {len(operations)} operations")
            return True
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def revert_to(self, position):
        """
        Revert to a specific position in the timeline.
        
        Args:
            position (int): The position to revert to.
            
        Returns:
            list: Operations up to the specified position.
        """
        if not self.current_timeline:
            logger.warning("Cannot revert: Timeline is empty")
            return []
        
        if position < 0 or position >= len(self.current_timeline):
            logger.warning(f"Cannot revert: Position {position} is out of range")
            return []
        
        self.current_position = position
        logger.info(f"Reverted to position {position}: {self.current_timeline[position].description}")
        
        return self.current_timeline[:position + 1]
    
    def move_back(self, steps=1):
        """
        Move back in the timeline.
        
        Args:
            steps (int): Number of steps to move back.
            
        Returns:
            Operation or None: The operation at the new position, or None if out of range.
        """
        new_position = self.current_position - steps
        
        if new_position < -1:
            logger.warning(f"Cannot move back {steps} steps: Out of range")
            return None
        
        self.current_position = new_position
        
        if new_position == -1:
            logger.info("Moved back to beginning of timeline")
            return None
        else:
            logger.info(f"Moved back to position {new_position}: {self.current_timeline[new_position].description}")
            return self.current_timeline[new_position]
    
    def move_forward(self, steps=1):
        """
        Move forward in the timeline.
        
        Args:
            steps (int): Number of steps to move forward.
            
        Returns:
            Operation or None: The operation at the new position, or None if out of range.
        """
        new_position = self.current_position + steps
        
        if new_position >= len(self.current_timeline):
            logger.warning(f"Cannot move forward {steps} steps: Out of range")
            return None
        
        self.current_position = new_position
        logger.info(f"Moved forward to position {new_position}: {self.current_timeline[new_position].description}")
        
        return self.current_timeline[new_position]
    
    def get_current_operation(self):
        """
        Get the current operation.
        
        Returns:
            Operation or None: The current operation, or None if at the beginning of the timeline.
        """
        if self.current_position == -1 or not self.current_timeline:
            return None
        
        return self.current_timeline[self.current_position]
    
    def get_operations_by_type(self, operation_type, session_id):
        """
        Get operations of a specific type.
        
        Args:
            operation_type (str): The operation type to filter by.
            session_id (str): The session ID to get operations for.
            
        Returns:
            list: List of operations of the specified type.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM operations WHERE operation_type = ? AND session_id = ? ORDER BY timestamp",
                (operation_type, session_id)
            )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Retrieved {len(operations)} operations of type {operation_type}")
            return operations
        except Exception as e:
            logger.error(f"Error retrieving operations by type: {e}")
            return []
    
    def search_operations(self, search_term, session_id):
        """
        Search operations by description.
        
        Args:
            search_term (str): The term to search for.
            session_id (str): The session ID to search in.
            
        Returns:
            list: List of matching operations.
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM operations WHERE description LIKE ? AND session_id = ? ORDER BY timestamp",
                (f"%{search_term}%", session_id)
            )
            
            operations = []
            for row in cursor.fetchall():
                operation = Operation(
                    operation_type=row[2],
                    parameters=json.loads(row[3]),
                    description=row[4]
                )
                operation.id = row[0]
                operation.timestamp = row[1]
                operations.append(operation)
            
            conn.close()
            
            logger.info(f"Found {len(operations)} operations matching '{search_term}'")
            return operations
        except Exception as e:
            logger.error(f"Error searching operations: {e}")
            return []
EOF

# Create nlp_processor module files
cat > src/nlp_processor/__init__.py << 'EOF'
"""
NLP processor module for interpreting CAD commands.
This module provides functionality for processing natural language CAD commands.
"""

from .command_processor import CommandProcessor

__all__ = ['CommandProcessor']
EOF

cat > src/nlp_processor/command_processor.py << 'EOF'
"""
Command processor module for interpreting CAD commands.
This module processes natural language commands for CAD operations.
"""

import os
import re
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CommandProcessor:
    """
    Processes natural language commands for CAD operations.
    Uses AI models to interpret commands and extract parameters.
    """
    
    def __init__(self, use_gpt=True, model="gpt-4o"):
        """
        Initialize the command processor.
        
        Args:
            use_gpt (bool): Whether to use GPT for processing.
            model (str): The GPT model to use.
        """
        self.use_gpt = use_gpt
        self.model = model
        
        # Initialize OpenAI client if using GPT
        if self.use_gpt:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found in environment variables")
            else:
                openai.api_key = api_key
                logger.info(f"OpenAI client initialized with model: {self.model}")
        
        # Define command patterns
        self.command_patterns = {
            "sketch": r"(?i)sketch|draw|create\s+(?:a|an)?\s*(\w+)",
            "extrude": r"(?i)extrude|pull|push\s+(?:by|to)?\s*(\d+(?:\.\d+)?)",
            "revolve": r"(?i)revolve|rotate|spin\s+(?:by|through)?\s*(\d+(?:\.\d+)?)",
            "fillet": r"(?i)fillet|round|smooth\s+(?:with|by|radius)?\s*(\d+(?:\.\d+)?)",
            "mirror": r"(?i)mirror|reflect|flip\s+(?:across|over|on)?\s*(x|y|z|xy|xz|yz)",
            "timeline": r"(?i)(?:go|return|revert|move)\s+(?:back|to)\s+(\d+|\w+)"
        }
        
        logger.info("Command processor initialized")
    
    def basic_pattern_matching(self, command):
        """
        Process a command using basic pattern matching.
        
        Args:
            command (str): The command to process.
            
        Returns:
            dict: Processed command information.
        """
        for cmd_type, pattern in self.command_patterns.items():
            match = re.search(pattern, command)
            if match:
                param_value = match.group(1)
                logger.info(f"Matched command type: {cmd_type} with parameter: {param_value}")
                
                return {
                    "type": cmd_type,
                    "parameters": self._extract_parameters(cmd_type, param_value, command),
                    "original_command": command
                }
        
        logger.warning(f"No pattern match found for command: {command}")
        return {
            "type": "unknown",
            "parameters": {},
            "original_command": command
        }
    
    def _extract_parameters(self, cmd_type, param_value, full_command):
        """
        Extract parameters from a command based on its type.
        
        Args:
            cmd_type (str): The type of command.
            param_value (str): The parameter value from regex match.
            full_command (str): The full command string.
            
        Returns:
            dict: Extracted parameters.
        """
        if cmd_type == "sketch":
            # Extract shape and dimensions
            shape = param_value.lower()
            dimensions = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x|by|mm|cm|m)?', full_command)
            
            params = {
                "shape": shape
            }
            
            if dimensions:
                if len(dimensions) == 1:
                    # Single dimension (square, circle)
                    if shape in ["circle", "sphere"]:
                        params["radius"] = float(dimensions[0])
                    else:
                        params["size"] = float(dimensions[0])
                elif len(dimensions) >= 2:
                    # Multiple dimensions (rectangle)
                    params["width"] = float(dimensions[0])
                    params["height"] = float(dimensions[1])
                    if len(dimensions) > 2 and shape in ["cube", "box"]:
                        params["depth"] = float(dimensions[2])
            
            return params
            
        elif cmd_type == "extrude":
            # Extract extrusion distance
            try:
                distance = float(param_value)
                
                # Check for "up", "down", etc. to determine direction
                direction = "up"  # Default
                if re.search(r'(?i)down|negative|below', full_command):
                    direction = "down"
                
                return {
                    "distance": distance,
                    "direction": direction
                }
            except ValueError:
                return {"distance": 10.0, "direction": "up"}  # Default
                
        elif cmd_type == "revolve":
            # Extract rotation angle
            try:
                angle = float(param_value)
                
                # Check for axis information
                axis = "z"  # Default
                if re.search(r'(?i)x\s*axis', full_command):
                    axis = "x"
                elif re.search(r'(?i)y\s*axis', full_command):
                    axis = "y"
                
                return {
                    "angle": angle,
                    "axis": axis
                }
            except ValueError:
                return {"angle": 360.0, "axis": "z"}  # Default
                
        elif cmd_type == "fillet":
            # Extract fillet radius
            try:
                radius = float(param_value)
                return {"radius": radius}
            except ValueError:
                return {"radius": 1.0}  # Default
                
        elif cmd_type == "mirror":
            # Extract mirror plane
            plane = param_value.lower()
            
            # Map text to standard planes
            plane_mapping = {
                "x": "yz", "y": "xz", "z": "xy",
                "xy": "xy", "xz": "xz", "yz": "yz"
            }
            
            return {"plane": plane_mapping.get(plane, "xy")}
            
        elif cmd_type == "timeline":
            # Handle timeline navigation
            try:
                # Check if it's a step number
                steps = int(param_value)
                return {"steps": steps}
            except ValueError:
                # It's a description
                return {"description": param_value}
        
        return {}
    
    def process_with_gpt(self, command):
        """
        Process a command using GPT.
        
        Args:
            command (str): The command to process.
            
        Returns:
            dict: Processed command information.
        """
        if not self.use_gpt or not openai.api_key:
            logger.warning("GPT processing not available, falling back to pattern matching")
            return self.basic_pattern_matching(command)
        
        try:
            # Define system prompt for processing CAD commands
            system_prompt = """
            You are an AI assistant specialized in interpreting natural language CAD commands.
            Your task is to take a natural language command and convert it into a structured format.
            
            You should extract the following information:
            1. Command type (sketch, extrude, revolve, fillet, mirror, timeline)
            2. All relevant parameters with values
            3. Units (if specified)
            
            Respond in JSON format with the following structure:
            {
                "type": "command_type",
                "parameters": {
                    "param1": value1,
                    "param2": value2,
                    ...
                },
                "original_command": "the original command"
            }
            
            Common command types and their parameters:
            - sketch: shape, size/width/height/radius
            - extrude: distance, direction
            - revolve: angle, axis
            - fillet: radius
            - mirror: plane (xy, xz, yz)
            - timeline: steps or description
            
            Make reasonable assumptions for any missing parameters.
            Respond ONLY with the JSON object, no additional text.
            """
            
            # Send request to OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract and parse JSON from response
            result_text = response.choices[0].message.content
            try:
                result = json.loads(result_text)
                logger.info(f"GPT processed command: {command} → {result['type']}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse GPT response as JSON: {result_text}")
                # Fall back to pattern matching
                return self.basic_pattern_matching(command)
                
        except Exception as e:
            logger.error(f"Error processing command with GPT: {e}")
            # Fall back to pattern matching
            return self.basic_pattern_matching(command)
    
    def process_command(self, command):
        """
        Process a natural language command.
        
        Args:
            command (str): The command to process.
            
        Returns:
            dict: Processed command information.
        """
        # Check for empty/null command
        if not command:
            logger.warning("Received empty command")
            return {
                "type": "unknown",
                "parameters": {},
                "original_command": command
            }
        
        # Remove any punctuation
        command = re.sub(r'[^\w\s]', ' ', command)
        
        # Process with GPT if available, otherwise use pattern matching
        if self.use_gpt and openai.api_key:
            return self.process_with_gpt(command)
        else:
            return self.basic_pattern_matching(command)
EOF

# Create main.py
cat > src/main.py << 'EOF'
"""
Main entry point for OpenSCAD Voice Control application.
This file contains the main application logic and serves as the entry point.
"""

import os
import sys
import logging
import time
import uuid
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
import click
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from voice_recognition import VoiceRecognizer, MicrophoneManager
from nlp_processor import CommandProcessor
from openscad_interface import OpenSCADCodeGenerator, OpenSCADRenderer
from timeline import TimelineManager, Operation

# Configure rich console for better terminal output
console = Console()

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("speech-to-cad")

class SpeechToCADApp:
    """Main application class for Speech-to-CAD."""
    
    def __init__(self, use_google=False, use_whisper=False, device_index=None):
        """
        Initialize the application.
        
        Args:
            use_google (bool): Whether to use Google Speech Recognition.
            use_whisper (bool): Whether to use OpenAI Whisper API.
            device_index (int, optional): Microphone device index.
        """
        console.print("[bold blue]Initializing Speech-to-CAD Application...[/bold blue]")
        
        # Initialize components
        self.mic_manager = MicrophoneManager(device_index=device_index)
        self.recognizer = VoiceRecognizer(use_google=use_google, use_whisper=use_whisper)
        self.command_processor = CommandProcessor(use_gpt=True)
        self.code_generator = OpenSCADCodeGenerator()
        self.renderer = OpenSCADRenderer()
        self.timeline_manager = TimelineManager()
        
        # Create a session
        self.session_id = self.timeline_manager.create_session()
        
        # Set up directories
        self.models_dir = Path("../models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        console.print("[bold green]Application initialized successfully![/bold green]")
    
    def listen_for_command(self):
        """
        Listen for a voice command.
        
        Returns:
            str: The recognized command, or None if not recognized.
        """
        try:
            # Get microphone source
            mic_source = self.mic_manager.start_listening()
            
            # Recognize speech
            console.print("[yellow]Listening for command...[/yellow]")
            text = self.recognizer.recognize_from_microphone(mic_source)
            
            if text:
                console.print(f"[green]Recognized:[/green] [bold]{text}[/bold]")
                return text
            else:
                console.print("[red]Could not recognize speech[/red]")
                return None
        except Exception as e:
            console.print(f"[bold red]Error during listening:[/bold red] {str(e)}")
            return None
    
    def process_command(self, command_text):
        """
        Process a command.
        
        Args:
            command_text (str): The command text to process.
            
        Returns:
            dict: The processed command.
        """
        try:
            # Process the command
            command = self.command_processor.process_command(command_text)
            
            # Display the processed command
            if command["type"] != "unknown":
                console.print(f"[green]Command type:[/green] [bold]{command['type']}[/bold]")
                console.print("[green]Parameters:[/green]")
                for key, value in command["parameters"].items():
                    console.print(f"  [cyan]{key}:[/cyan] {value}")
            else:
                console.print("[red]Unknown command type[/red]")
            
            return command
        except Exception as e:
            console.print(f"[bold red]Error processing command:[/bold red] {str(e)}")
            return {"type": "unknown", "parameters": {}, "original_command": command_text}
    
    def execute_command(self, command):
        """
        Execute a processed command.
        
        Args:
            command (dict): The processed command.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            command_type = command["type"]
            parameters = command["parameters"]
            
            if command_type == "unknown":
                console.print("[red]Cannot execute unknown command[/red]")
                return False
            
            # Execute based on command type
            if command_type == "sketch":
                return self._execute_sketch_command(parameters)
            elif command_type == "extrude":
                return self._execute_extrude_command(parameters)
            elif command_type == "revolve":
                return self._execute_revolve_command(parameters)
            elif command_type == "fillet":
                return self._execute_fillet_command(parameters)
            elif command_type == "mirror":
                return self._execute_mirror_command(parameters)
            elif command_type == "timeline":
                return self._execute_timeline_command(parameters)
            else:
                console.print(f"[red]Unsupported command type: {command_type}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error executing command:[/bold red] {str(e)}")
            return False
    
    def _execute_sketch_command(self, parameters):
        """Execute a sketch command."""
        shape = parameters.get("shape", "square")
        
        try:
            # Create the shape based on type
            if shape == "circle":
                radius = parameters.get("radius", 10.0)
                result = self.code_generator.create_circle(radius=radius, name=f"circle_{uuid.uuid4().hex[:8]}")
            elif shape in ["square", "rectangle"]:
                width = parameters.get("width", parameters.get("size", 10.0))
                height = parameters.get("height", width)
                result = self.code_generator.create_square(
                    size=[width, height] if width != height else width,
                    center=True,
                    name=f"{shape}_{uuid.uuid4().hex[:8]}"
                )
            else:
                console.print(f"[red]Unsupported shape: {shape}[/red]")
                return False
            
            # Add to timeline
            if result:
                operation = Operation(
                    operation_type="sketch",
                    parameters=parameters,
                    description=f"Sketched a {shape}"
                )
                self.timeline_manager.add_operation(operation, self.session_id)
                
                # Preview the result
                self._preview_current_model()
                
                return True
            else:
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error executing sketch command:[/bold red] {str(e)}")
            return False
    
    def _execute_extrude_command(self, parameters):
        """Execute an extrude command."""
        distance = parameters.get("distance", 10.0)
        direction = parameters.get("direction", "up")
        
        try:
            # Get the current model or shape to extrude
            if self.code_generator.current_model is None:
                console.print("[red]No model to extrude[/red]")
                return False
            
            # Adjust distance based on direction
            if direction == "down":
                distance = -distance
            
            # Perform the extrusion
            result = self.code_generator.extrude_linear(
                shape=self.code_generator.current_model,
                height=distance,
                center=False,
                name=f"extrusion_{uuid.uuid4().hex[:8]}"
            )
            
            # Add to timeline
            if result:
                operation = Operation(
                    operation_type="extrude",
                    parameters=parameters,
                    description=f"Extruded by {distance}mm"
                )
                self.timeline_manager.add_operation(operation, self.session_id)
                
                # Preview the result
                self._preview_current_model()
                
                return True
            else:
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error executing extrude command:[/bold red] {str(e)}")
            return False
    
    def _execute_revolve_command(self, parameters):
        """Execute a revolve command."""
        angle = parameters.get("angle", 360.0)
        axis = parameters.get("axis", "z")
        
        try:
            # Get the current model or shape to revolve
            if self.code_generator.current_model is None:
                console.print("[red]No model to revolve[/red]")
                return False
            
            # Perform the revolve
            result = self.code_generator.extrude_rotate(
                shape=self.code_generator.current_model,
                angle=angle,
                name=f"revolution_{uuid.uuid4().hex[:8]}"
            )
            
            # Add to timeline
            if result:
                operation = Operation(
                    operation_type="revolve",
                    parameters=parameters,
                    description=f"Revolved around {axis} axis by {angle} degrees"
                )
                self.timeline_manager.add_operation(operation, self.session_id)
                
                # Preview the result
                self._preview_current_model()
                
                return True
            else:
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error executing revolve command:[/bold red] {str(e)}")
            return False
    
    def _execute_mirror_command(self, parameters):
        """Execute a mirror command."""
        plane = parameters.get("plane", "xy")
        
        try:
            # Get the current model to mirror
            if self.code_generator.current_model is None:
                console.print("[red]No model to mirror[/red]")
                return False
            
            # Convert plane to vector
            vector_mapping = {
                "xy": [0, 0, 1],
                "xz": [0, 1, 0],
                "yz": [1, 0, 0]
            }
            vector = vector_mapping.get(plane, [0, 0, 1])
            
            # Perform the mirror
            result = self.code_generator.mirror(
                shape=self.code_generator.current_model,
                vector=vector,
                name=f"mirror_{uuid.uuid4().hex[:8]}"
            )
            
            # Add to timeline
            if result:
                operation = Operation(
                    operation_type="mirror",
                    parameters=parameters,
                    description=f"Mirrored across {plane} plane"
                )
                self.timeline_manager.add_operation(operation, self.session_id)
                
                # Preview the result
                self._preview_current_model()
                
                return True
            else:
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error executing mirror command:[/bold red] {str(e)}")
            return False
    
    def _execute_fillet_command(self, parameters):
        """Execute a fillet command."""
        # Note: Filleting in OpenSCAD requires more complex operations
        # For now, we'll just log the command and add it to the timeline
        radius = parameters.get("radius", 1.0)
        
        console.print("[yellow]Fillet operation not fully implemented yet[/yellow]")
        
        # Add to timeline
        operation = Operation(
            operation_type="fillet",
            parameters=parameters,
            description=f"Applied fillet with radius {radius}mm"
        )
        self.timeline_manager.add_operation(operation, self.session_id)
        
        return True
    
    def _execute_timeline_command(self, parameters):
        """Execute a timeline command."""
        if "steps" in parameters:
            # Move back a specified number of steps
            steps = parameters["steps"]
            result = self.timeline_manager.move_back(steps)
            
            if result is not None:
                console.print(f"[green]Moved back {steps} steps to: {result.description}[/green]")
                
                # Rebuild the model up to this point
                self._rebuild_model_to_current_position()
                
                return True
            else:
                console.print("[red]Could not move back in timeline[/red]")
                return False
                
        elif "description" in parameters:
            # Find operations matching description
            search_term = parameters["description"]
            operations = self.timeline_manager.search_operations(search_term, self.session_id)
            
            if operations:
                # Find the index of the first matching operation
                position = 0
                for i, op in enumerate(self.timeline_manager.current_timeline):
                    if op.id == operations[0].id:
                        position = i
                        break
                
                # Revert to that position
                self.timeline_manager.revert_to(position)
                console.print(f"[green]Reverted to: {operations[0].description}[/green]")
                
                # Rebuild the model
                self._rebuild_model_to_current_position()
                
                return True
            else:
                console.print(f"[red]No operations found matching '{search_term}'[/red]")
                return False
        else:
            console.print("[red]Invalid timeline command parameters[/red]")
            return False
    
    def _preview_current_model(self):
        """Preview the current model in OpenSCAD."""
        try:
            # Generate the code
            code = self.code_generator.generate_code()
            
            if code:
                # Save to a file
                filename = os.path.join(self.models_dir, f"model_{int(time.time())}.scad")
                self.code_generator.save_code(filename)
                
                # Preview in OpenSCAD
                self.renderer.preview(code)
                
                console.print(f"[green]Model previewed and saved to {filename}[/green]")
                return True
            else:
                console.print("[red]No model to preview[/red]")
                return False
                
        except Exception as e:
            console.print(f"[bold red]Error previewing model:[/bold red] {str(e)}")
            return False
    
    def _rebuild_model_to_current_position(self):
        """Rebuild the model up to the current timeline position."""
        try:
            # Reset the code generator
            self.code_generator.reset()
            
            # Get operations up to current position
            operations = self.timeline_manager.current_timeline[:self.timeline_manager.current_position + 1]
            
            if not operations:
                console.print("[yellow]No operations to rebuild[/yellow]")
                return
            
            # Re-execute each operation
            for op in operations:
                console.print(f"[cyan]Rebuilding: {op.description}[/cyan]")
                
                # Execute based on operation type
                if op.operation_type == "sketch":
                    self._execute_sketch_command(op.parameters)
                elif op.operation_type == "extrude":
                    self._execute_extrude_command(op.parameters)
                elif op.operation_type == "revolve":
                    self._execute_revolve_command(op.parameters)
                elif op.operation_type == "mirror":
                    self._execute_mirror_command(op.parameters)
                elif op.operation_type == "fillet":
                    # Fillet not fully implemented yet
                    pass
            
            # Preview the rebuilt model
            self._preview_current_model()
            
        except Exception as e:
            console.print(f"[bold red]Error rebuilding model:[/bold red] {str(e)}")
    
    def show_timeline(self):
        """Display the current timeline."""
        operations = self.timeline_manager.current_timeline
        
        if not operations:
            console.print("[yellow]Timeline is empty[/yellow]")
            return
        
        # Create a table to display operations
        table = Table(title="Operation Timeline")
        table.add_column("Position", style="cyan")
        table.add_column("Timestamp", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Description", style="white")
        
        for i, op in enumerate(operations):
            marker = "→" if i == self.timeline_manager.current_position else ""
            table.add_row(
                f"{marker} {i}",
                op.readable_timestamp,
                op.operation_type,
                op.description
            )
        
        console.print(table)
    
    def run_interactive(self):
        """Run the application in interactive mode."""
        try:
            # Display banner
            console.print(Panel(
                "[bold blue]OpenSCAD Voice Control[/bold blue]\n"
                "[italic]Say commands to create 3D models[/italic]",
                expand=False
            ))
            
            # Adjust for ambient noise
            console.print("\n[yellow]Adjusting for ambient noise... Please be quiet for a moment.[/yellow]")
            self.mic_manager.adjust_for_ambient_noise(duration=2)
            
            console.print("\n[bold green]Ready for voice commands![/bold green]")
            console.print("Say something like: [bold cyan]'Sketch a 10x10mm square'[/bold cyan]")
            console.print("Or press [bold]Ctrl+C[/bold] to exit\n")
            
            # Show help
            self._show_help()
            
            # Main command loop
            while True:
                console.print("\n[bold blue]Waiting for command...[/bold blue]")
                
                # Listen for command
                command_text = self.listen_for_command()
                
                if command_text:
                    # Process the command
                    command = self.process_command(command_text)
                    
                    # Check for special commands
                    if command_text.lower() in ["exit", "quit", "stop"]:
                        console.print("[bold]Exiting application...[/bold]")
                        break
                    elif command_text.lower() == "help":
                        self._show_help()
                        continue
                    elif command_text.lower() == "show timeline":
                        self.show_timeline()
                        continue
                    
                    # Execute the command
                    success = self.execute_command(command)
                    
                    if success:
                        console.print("[green]Command executed successfully[/green]")
                    else:
                        console.print("[red]Command execution failed[/red]")
                
                time.sleep(0.5)  # Short pause between commands
                
        except KeyboardInterrupt:
            console.print("\n[bold]Exiting OpenSCAD Voice Control...[/bold]")
        except Exception as e:
            logger.exception("An error occurred")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _show_help(self):
        """Display help information."""
        console.print(Panel(
            "[bold]Available Commands:[/bold]\n"
            "- [cyan]Sketch a 10x10mm square[/cyan] - Create a square\n"
            "- [cyan]Sketch a circle with 5mm radius[/cyan] - Create a circle\n"
            "- [cyan]Extrude by 20mm[/cyan] - Extrude the current shape\n"
            "- [cyan]Revolve around Z axis[/cyan] - Create a revolved shape\n"
            "- [cyan]Mirror across XY plane[/cyan] - Mirror the current shape\n"
            "- [cyan]Go back 2 steps[/cyan] - Navigate timeline\n"
            "- [cyan]Show timeline[/cyan] - Display operation history\n"
            "- [cyan]Help[/cyan] - Show this help\n"
            "- [cyan]Exit[/cyan] - Quit the application",
            title="Help",
            expand=False
        ))

@click.command()
@click.option('--use-google', is_flag=True, help='Use Google Speech Recognition instead of default')
@click.option('--use-whisper', is_flag=True, help='Use OpenAI Whisper API for speech recognition')
@click.option('--device-index', type=int, help='Microphone device index to use')
@click.option('--list-devices', is_flag=True, help='List available microphone devices')
def main(use_google, use_whisper, device_index, list_devices):
    """OpenSCAD Voice Control - Convert speech to CAD operations."""
    
    try:
        # List devices if requested
        if list_devices:
            mic_manager = MicrophoneManager()
            devices = mic_manager.list_microphone_devices()
            console.print("\n[bold]Available Microphone Devices:[/bold]")
            for device in devices:
                console.print(f"  [bold cyan]Index {device['index']}:[/bold cyan] {device['name']}")
            return
        
        # Initialize and run the application
        app = SpeechToCADApp(
            use_google=use_google,
            use_whisper=use_whisper,
            device_index=device_index
        )
        app.run_interactive()
            
    except KeyboardInterrupt:
        console.print("\n[bold]Exiting OpenSCAD Voice Control...[/bold]")
    except Exception as e:
        logger.exception("An error occurred")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
EOF

# Create .env file
cat > .env << 'EOF'
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Application Settings
DEBUG=True

# OpenSCAD Settings
OPENSCAD_PATH=/path/to/openscad
EOF

echo "Project structure created successfully!"