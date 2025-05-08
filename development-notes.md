# Development Notes: OpenSCAD Voice Control

## Project Requirements

### Core Functionality
- Voice-to-text transcription for CAD commands
- Natural language processing to interpret CAD terminology
- Conversion of commands into OpenSCAD code
- Integration with OpenSCAD for rendering

### CAD Operations
- **Sketch Operations**: Create 2D shapes like squares, circles, rectangles
- **Extrusion Operations**: Extrude 2D shapes into 3D objects
- **Revolve Operations**: Create revolved shapes around an axis
- **Fillet Operations**: Add rounded edges to models
- **Mirror Operations**: Create mirrored copies of objects

### Timeline System
- Global timeline tracking all operations in chronological order
- Feature-specific timelines for individual CAD features
- Support for navigation commands:
  - "Return to when the first extrude was 20mm"
  - "Go back 3 steps"
  - "Restore the model from earlier version"
- Parameter history tracking for each feature

### User Experience
- Natural voice interaction similar to ChatGPT
- Simple, intuitive commands using CAD terminology
- Clear feedback on operation success/failure

## Technical Decisions
- **Language**: Python 3 (compatible with Python 3.13)
- **Voice Recognition**: SpeechRecognition library
- **AI Models**: Initially use API access to Claude/GPT for rapid development
- **OpenSCAD Interface**: SolidPython for generating OpenSCAD code
- **Database**: SQLite for timeline storage

## Example Commands
- "Sketch a 10x10mm square"
- "Extrude that square 20mm up"
- "Fillet the edges with 2mm radius"
- "Mirror this part across the XZ plane"
- "Return to when the extrusion was 15mm tall"

## Future Expansion
- Web application interface
- Desktop application
- Custom AI model for CAD-specific terminology
