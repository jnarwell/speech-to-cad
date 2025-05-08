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
            "sketch": r"(?i)(sketch|draw|create)\s+(?:a|an)?\s*(square|circle|rectangle)?",
            "extrude": r"(?i)extrude\s+(?:by|to)?\s*(\d+(?:\.\d+)?)",
            "revolve": r"(?i)revolve\s+(?:by|through)?\s*(\d+(?:\.\d+)?)",
            "fillet": r"(?i)fillet\s+(?:with|by|radius)?\s*(\d+(?:\.\d+)?)",
            "mirror": r"(?i)mirror\s+(?:across|over|on)?\s*(x|y|z|xy|xz|yz)",
            "timeline": r"(?i)(?:go|return|revert|move)\s+(?:back|to)\s+(\d+|\w+)",
            "exit": r"(?i)^(exit|quit|stop)$",
            "help": r"(?i)^help$",
            "timeline_show": r"(?i)^show\s+timeline$"
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
        
        Args:
            cmd_type (str): The type of command.
            param_value (str): The parameter value from regex match.
            full_command (str): The full command string.
            
        Returns:
            dict: Extracted parameters.
        """
        # Special commands that don't need parameters
        if cmd_type in ["exit", "help", "timeline_show"]:
            return {}
        
        if cmd_type == "sketch":
            # Look for shape name in the full command
            shape_match = re.search(r'(?i)(square|circle|rectangle)', full_command)
            shape = "square"  # Default shape
            
            if shape_match:
                shape = shape_match.group(1).lower()
            
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
                    if len(dimensions) > 2 and shape in ["cube", "box"]:
                        params["depth"] = float(dimensions[2])
            
            return params
            
        elif cmd_type == "extrude":
            # Extract extrusion distance
            try:
                distance = float(param_value) if param_value else 10.0
                
                # Check for "up", "down", etc. to determine direction
                direction = "up"  # Default
                if re.search(r'(?i)down|negative|below', full_command):
                    direction = "down"
                
                return {
                    "distance": distance,
                    "direction": direction
                }
            except (ValueError, TypeError):
                return {"distance": 10.0, "direction": "up"}  # Default
                
        elif cmd_type == "revolve":
            # Extract rotation angle
            try:
                angle = float(param_value) if param_value else 360.0
                
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
            except (ValueError, TypeError):
                return {"angle": 360.0, "axis": "z"}  # Default
                
        elif cmd_type == "fillet":
            # Extract fillet radius
            try:
                radius = float(param_value) if param_value else 1.0
                return {"radius": radius}
            except (ValueError, TypeError):
                return {"radius": 1.0}  # Default
                
        elif cmd_type == "mirror":
            # Extract mirror plane
            plane = param_value.lower() if param_value else "xy"
            
            # Map text to standard planes
            plane_mapping = {
                "x": "yz", "y": "xz", "z": "xy",
                "xy": "xy", "xz": "xz", "yz": "yz"
            }
            
            return {"plane": plane_mapping.get(plane, "xy")}
            
        elif cmd_type == "timeline":
            # Handle timeline navigation
            if not param_value:
                return {"steps": 1}  # Default to one step back
                
            try:
                # Check if it's a step number
                steps = int(param_value)
                return {"steps": steps}
            except ValueError:
                # It's a description
                return {"description": param_value}
        
        # Default empty parameters for unknown command types
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
            - exit: (no parameters, used to exit the application)
            - help: (no parameters, used to display help information)
            
            Make reasonable assumptions for any missing parameters.
            Only return valid JSON without code blocks or other formatting.
            """
            
            # Send request to OpenAI API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ]
            
            completion = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract and parse JSON from response
            result_text = completion.choices[0].message.content
            
            # Remove markdown code block formatting if present
            if "```json" in result_text:
                result_text = result_text.replace("```json", "", 1)
                if "```" in result_text:
                    result_text = result_text.rsplit("```", 1)[0]
                result_text = result_text.strip()
                
            try:
                result = json.loads(result_text)
                logger.info(f"GPT processed command: {command} → {result['type']}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse GPT response as JSON: {result_text}")
                # Fall back to pattern matching
                return self.basic_pattern_matching(command)
                    
        except Exception as e:
            logger.error(f"Error processing command with GPT: {str(e)}")
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
