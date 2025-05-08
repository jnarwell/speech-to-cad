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
