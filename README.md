# AI Suite for Startup Workflow Optimization

A hierarchical AI system with shared memory designed to optimize workflow for startups. This suite enables founders to delegate tasks to specialized AI assistants while maintaining a single command interface.

## Overview

This AI Suite features a hierarchical command structure with:

- **CEO (You)**: Issue high-level instructions to the system
- **Manager Node**: Central coordinator that delegates tasks to specialized nodes
- **Specialized Nodes**: Handle specific functions like marketing, research, etc.

All nodes share a common memory system allowing for consistent information sharing and preventing repetition.

## Key Features

- **Hierarchical Command Structure**: Maintains proper operation order
- **Shared Memory System**: Allows all AI nodes to access the same information
- **Single Folder Implementation**: All components live in one folder structure
- **Specialized AI Nodes**: Each optimized for specific tasks

## Getting Started

### Installation

1. Clone this repository to your project folder
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys:
   - Create a `.env` file with your API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     CLAUDE_API_KEY=your_claude_api_key
     ```
   - Or export them as environment variables

### Basic Usage

Run the AI Suite with a command:

```bash
python main.py --instruction "Create a pitch for my SaaS product that helps restaurants manage deliveries"
```

Specify a project:

```bash
python main.py --instruction "Research competitors in the restaurant delivery space" --project "food_delivery_app"
```

Check system status:

```bash
python main.py --status
```

## Specialized Nodes

### Brand Advisor

Generates brand guidelines, colors, taglines, and visual identity elements.

**Example:** `"Create a brand identity for my AI-powered restaurant delivery management app"`

### Dev Log Processor

Converts development logs into engaging social media posts.

**Example:** `"Turn today's dev log into a Twitter post"`

### Post Creator

Creates and schedules finalized social media posts across platforms.

**Example:** `"Create a LinkedIn post about our latest feature launch"`

### Researcher

Performs market and technical research on specified topics.

**Example:** `"Research the current state of AI in restaurant delivery management"`

### Pitch Generator

Creates sales pitches, investor presentations, and demo scripts.

**Example:** `"Generate an elevator pitch for potential investors"`

### Social Handler

Manages social media interactions and responds to comments.

**Example:** `"Create responses to recent Twitter comments about our product"`

## Directory Structure

```
/ai-suite/
  |-- main.py              # Entry point
  |-- memory/              # Shared memory storage
  |-- config/              # Configuration files
  |-- logs/                # System logs
  |-- data/                # Working data directory
      |-- dev_logs/        # Development logs
      |-- projects/        # Project-specific data
```

## Shared Memory System

The AI Suite uses a shared memory system stored in JSON format. This allows all AI nodes to access the same information, including:

- Project details
- Brand information
- Research findings
- Generated content
- Interaction history

## Advanced Features

### Project Management

Define multiple projects and switch between them:

```bash
python main.py --instruction "Update brand colors" --project "project_name"
```

### Automatic Dev Log Processing

Place your development logs in the `data/dev_logs` folder, and the system will automatically process them:

```bash
python main.py --instruction "Process latest dev log for social media"
```

### Chained Operations

The system supports chained operations where output from one node becomes input for another:

```bash
python main.py --instruction "Research competitors, then create a pitch highlighting our unique advantages"
```

## Extending the System

### Adding New AI Nodes

1. Create a new class that inherits from `AINode`
2. Implement the `process` method
3. Add the node type to `RoleType` enum
4. Register the node with the manager in `AIOrchestrator._initialize_nodes`

### Custom Prompts

Modify the prompts in each node's `process` method to customize the AI behavior.

## Troubleshooting

### API Key Issues

If you encounter API errors, check that your API keys are properly set in the environment or `.env` file.

### Memory Corruption

If the shared memory becomes corrupted, delete the `memory/shared_memory.json` file and restart the system.

### Log Analysis

Check the `logs/system_log.txt` file for detailed operation logs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
