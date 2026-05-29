# AstraMind AI

> Intelligent AI Assistant with Reasoning, Memory, and Multi-Agent Capabilities

## Overview

AstraMind AI is a comprehensive AI assistant framework built in Python that combines advanced reasoning, persistent memory, and a multi-agent architecture to deliver intelligent, context-aware responses.

## Features

- **Multi-Step Reasoning**: Chain-of-thought reasoning with self-reflection and correction
- **Persistent Memory**: Short-term and long-term memory with importance-based retention and decay
- **Multi-Agent System**: Specialized agents for different task types (core, decision, planner)
- **Tool Integration**: Built-in tools for calculations, web search, file reading, and data analysis
- **REST API**: FastAPI-based API with OpenAI-compatible endpoints
- **Interactive UI**: Streamlit-based chat interface and monitoring dashboard

## Project Structure

```
astramind-ai/
├── core/               # Core engine, reasoning, memory, and config
├── models/             # Model loading, inference, and tokenization
├── agents/             # Multi-agent system (core, decision, planner)
├── tools/              # Tool implementations (calculator, search, etc.)
├── api/                # FastAPI server, routes, and schemas
├── frontend/           # Streamlit UI and dashboard
├── prompts/            # System and reasoning prompt templates
├── data/               # Memory storage and logs
├── tests/              # Unit and integration tests
├── main.py             # Main entry point
├── requirements.txt    # Python dependencies
└── .env.example        # Environment configuration template
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/astramind-ai.git
cd astramind-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Key settings:
#   MODEL_NAME - Hugging Face model to use
#   DEVICE - Device for inference (auto/cpu/cuda)
#   API_PORT - Port for the API server
```

### 3. Run

**Interactive CLI Mode:**
```bash
python main.py interactive
```

**API Server Mode:**
```bash
python main.py api
```

**Streamlit Chat UI:**
```bash
python main.py chat
```

**Streamlit Dashboard:**
```bash
python main.py dashboard
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/` | POST | Send a chat message |
| `/api/v1/chat/completions` | POST | OpenAI-compatible completions |
| `/api/v1/memory/store` | POST | Store a memory entry |
| `/api/v1/memory/search` | POST | Search memories |
| `/api/v1/tools/execute` | POST | Execute a tool directly |
| `/api/v1/tools/list` | GET | List available tools |
| `/api/v1/system/status` | GET | Engine status |
| `/api/v1/system/health` | GET | Health check |

## Agents

| Agent | Description |
|-------|-------------|
| **Astra Core** | General-purpose assistant for standard queries |
| **Decision** | Multi-criteria decision analysis and evaluation |
| **Planner** | Complex planning with multi-step execution |

## Tools

| Tool | Description |
|------|-------------|
| **Calculator** | Mathematical expressions and computations |
| **Web Search** | Internet search for current information |
| **File Reader** | Read and parse text, CSV, JSON files |
| **Data Analyzer** | Statistical analysis and trend detection |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_engine.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for the full list of available settings.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | meta-llama/Llama-3-8B-Instruct | Model identifier |
| `DEVICE` | auto | Inference device |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `MAX_TOKENS` | 2048 | Max generation tokens |
| `API_PORT` | 8000 | API server port |
| `DEBUG` | false | Debug mode |

## License

MIT License
