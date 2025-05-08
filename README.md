# OpenSCAD Voice Control

A voice-controlled 3D modeling application that translates natural language into OpenSCAD commands.

## Overview

This application allows users to create engineering-level 3D models by speaking commands into a microphone. The system interprets natural language instructions and converts them into OpenSCAD code, supporting operations like:

- Sketching (2D shapes)
- Extrusion
- Revolving
- Filleting
- Mirroring
- And more...

The application also features a comprehensive timeline system that tracks all operations and allows users to navigate model history using voice commands.

## Requirements

- Python 3.8+
- OpenSCAD
- Microphone access

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YOUR-USERNAME/openscad-voice.git
   cd openscad-voice
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys for voice recognition and AI services (instructions in `docs/setup.md`)

## Usage

Run the application with:

```
python src/main.py
```

See `docs/usage.md` for a full list of supported voice commands and features.

## Project Structure

```
openscad-voice/
├── src/               # Source code
│   ├── cad_operations/ # CAD operation modules
│   ├── timeline/       # Timeline tracking functionality
│   └── utils/          # Utility functions
├── models/            # OpenSCAD models
├── database/          # Timeline database
├── tests/             # Test files
└── docs/              # Documentation
```

## License

[MIT License](LICENSE)