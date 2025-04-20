# FEI Architecture Diagrams

This directory contains PlantUML diagrams that visualize the architecture of the FEI project.

## PlantUML Diagrams

The `puml` directory contains the following PlantUML diagrams:

- `fei_chat_sequence_diagram.puml`: Sequence diagram showing the chat interaction flow
- `fei_component_diagram.puml`: Component diagram showing the high-level components of the system
- `fei_core_class_diagram.puml`: Class diagram showing the core classes and their relationships
- `fei_evolution_architecture.puml`: Architecture diagram for the evolution system
- `fei_mcp_architecture.puml`: Architecture diagram for the Model-Controlled Process (MCP) system
- `fei_memory_architecture.puml`: Architecture diagram for the memory system
- `fei_tool_architecture.puml`: Architecture diagram for the tool system
- `fei_ui_architecture.puml`: Architecture diagram for the user interface system

## Viewing the Diagrams

To view these diagrams, you can use:

1. The PlantUML extension for VS Code
2. The PlantUML server at http://www.plantuml.com/plantuml/
3. The PlantUML command-line tool

## Generating PNG/SVG Files

To generate PNG or SVG files from the PlantUML diagrams, you can use the PlantUML command-line tool:

```bash
java -jar plantuml.jar docs/diagrams/puml/*.puml
```

This will generate PNG files in the same directory as the PUML files.
