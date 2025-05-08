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
        """
        for cmd_type, pattern in self.command_patterns.items():
            match = re.search(pattern, command)
            if match:
                # Handle special command types
                if cmd_type == "exit" or cmd_type == "help" or cmd_type == "timeline_show":
                    return {
                        "type": cmd_type,
                        "parameters": {},
                        "original_command": command
                    }
                
                param_value = match.group(1) if match.groups() else None
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
        """
        if cmd_type in ["exit", "help", "timeline_show"]:
            return {}
        
        if cmd_type == "sketch":
            # Look for shape in the full command
            shape = "square"  # Default
            shape_match = re.search(r"(?i)(square|circle|rectangle)", full_command)
            if shape_match:
                shape = shape_match.group(1).lower()
            elif param_value:
                # Use parameter from regex if available
                shape = param_value.lower()
            
            # Extract dimensions
            dimensions = re.findall(r'(\d+(?:\.\d+)?)\s*(?:x|by|mm|cm|m)?', full_command)
            
            params = {
                "shape": shape
            }
            
            if dimensions:
                if len(dimensions) == 1:
                    # Single dimension (square, circle)
                    if shape == "circle":
                        params["radius"] = float(dimensions[0])
                    else:
                        params["size"] = float(dimensions[0])
                elif len(dimensions) >= 2:
                    # Multiple dimensions (rectangle)
                    params["width"] = float(dimensions[0])
                    params["height"] = float(dimensions[1])
            
            return params
        
        # Keep the rest of the method as is...
    
    def process_with_gpt(self, command):
        # Extract and parse JSON from response
        result_text = response.choices[0].message.content
        try:
            # Remove markdown code block formatting if present
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "", 1)
                if result_text.endswith("```"):
                    result_text = result_text.rsplit("```", 1)[0]
                # Clean up any remaining whitespace
                result_text = result_text.strip()
                
            result = json.loads(result_text)
            logger.info(f"GPT processed command: {command} → {result['type']}")
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT response as JSON: {result_text}")
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
