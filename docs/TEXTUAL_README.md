# Modern Textual Chat Interface for FEI

## Overview

The FEI project now includes a modern terminal-based chat interface built with the [Textual](https://textual.textualize.io/) library. This interface provides a more visually appealing and interactive experience compared to the traditional command-line interface.

## Features

- **Rich Markdown Rendering**: Assistant responses are rendered as Markdown with syntax highlighting
- **Modern UI Components**: Message bubbles, panels, input box, and buttons
- **Visual Indicators**: Loading spinner while the assistant is generating a response
- **Keyboard Shortcuts**: Easy navigation and actions using keyboard shortcuts
- **Responsive Layout**: Automatically adapts to terminal size

## Usage

### Running the Textual Interface

You can run the Textual interface in two ways:

1. Using the `--textual` flag with the main command:
   ```bash
   fei --textual
   ```

2. Running the example script:
   ```bash
   python examples/textual_chat_example.py
   ```

### Command Line Options

The Textual interface supports all the same options as the traditional CLI:

```bash
# Use a specific provider and model
fei --textual --provider openai --model gpt-4o

# Enable debug logging
fei --textual --debug
```

### Keyboard Shortcuts

The following keyboard shortcuts are available in the Textual interface:

- `Ctrl+C`, `Ctrl+D`, or `Escape`: Quit the application
- `Ctrl+L`: Clear the chat history
- `Enter` (when input box is focused): Send the message

## Development

The Textual interface is implemented in `fei/ui/textual_chat.py`. It uses Textual's component-based architecture to create a responsive and interactive UI.

### Key Components

- `ChatMessage`: A custom widget for rendering user and assistant messages
- `FeiChatApp`: The main application class that handles the chat interface

### Styling

The interface uses Textual's CSS-like styling system for visual customization. The styles are defined inline in the `FeiChatApp` class.

## Dependencies

To use the Textual interface, you need to install the Textual library:

```bash
pip install textual>=0.47.1
```

This dependency is included in the project's `requirements.txt` file.