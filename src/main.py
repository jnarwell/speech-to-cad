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
