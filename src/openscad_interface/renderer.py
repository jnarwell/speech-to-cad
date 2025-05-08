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
