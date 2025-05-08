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
