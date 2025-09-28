# Evo AI Setup Guide

This guide walks you through setting up Evo AI from scratch, including all dependencies and configuration options.

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (3.11 recommended)
- **8GB+ RAM** (16GB recommended for learning)
- **Local LLM Server** (Ollama, LM Studio, or similar)

### Option 1: Automatic Setup
```bash
# Clone and run setup script
git clone https://github.com/your-repo/evo-ai.git
cd evo-ai
python scripts/setup.py
```

### Option 2: Manual Setup
Follow the detailed steps below for full control over the installation.

## 📋 Detailed Installation

### Step 1: Python Environment

#### Create Virtual Environment
```bash
# Using venv
python -m venv evo-env
source evo-env/bin/activate  # Linux/Mac
# or
evo-env\Scripts\activate     # Windows

# Using conda
conda create -n evo python=3.11
conda activate evo
```

#### Install Evo AI
```bash
# Install from source
git clone https://github.com/your-repo/evo-ai.git
cd evo-ai
pip install -e .

# Install development dependencies (optional)
pip install -e .[dev]
```

### Step 2: Local LLM Server Setup

Evo AI supports multiple local LLM servers:

#### Option A: Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh  # Linux/Mac
# or download from https://ollama.ai for Windows

# Pull the model
ollama pull llama3.2:7b

# Start Ollama (usually auto-starts)
ollama serve
```

#### Option B: LM Studio
1. Download LM Studio from https://lmstudio.ai
2. Install and launch LM Studio
3. Download a compatible model (Llama 3.2 7B recommended)
4. Start the local server (default: http://127.0.0.1:1234)

#### Option C: Text Generation WebUI
```bash
# Clone and setup
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
pip install -r requirements.txt

# Download model and start with API
python server.py --api --listen
```

### Step 3: Initialize Evo AI

```bash
# Initialize configuration
evo init

# For LM Studio users, update the config:
evo init --model-url http://127.0.0.1:1234 --model-name llama3.2
```

### Step 4: Verify Installation

```bash
# Run tests
pytest tests/

# Run demo
python scripts/run_demo.py

# Start interactive chat
evo chat
```

## ⚙️ Configuration

### Basic Configuration
Evo AI stores configuration in `~/.evo/config.json`:

```json
{
  "model_name": "llama3.2:7b",
  "model_url": "http://127.0.0.1:1234",
  "temperature": 0.7,
  "use_memory": true,
  "enable_learning": true,
  "enable_mcp": true,
  "database_path": "~/.evo/memory.db",
  "sessions_dir": "~/.evo/sessions"
}
```

### LM Studio Configuration
For LM Studio users, use this configuration:

```json
{
  "model_name": "llama3.2",
  "model_url": "http://127.0.0.1:1234",
  "model_type": "openai_compatible",
  "api_key": "lm-studio",
  "temperature": 0.7,
  "max_tokens": 4096,
  "use_memory": true,
  "enable_learning": true,
  "enable_mcp": true,
  "database_path": "~/.evo/memory.db",
  "sessions_dir": "~/.evo/sessions",
  "learning": {
    "enabled": true,
    "lora_rank": 16,
    "learning_rate": 0.0001,
    "training_interval_hours": 6,
    "importance_threshold": 0.5
  }
}
```

### Advanced Configuration

#### Memory Settings
```json
{
  "memory": {
    "embedding_model": "all-MiniLM-L6-v2",
    "max_episodic_memories": 10000,
    "consolidation_interval_hours": 12,
    "importance_threshold": 0.3,
    "enable_vector_search": true
  }
}
```

#### Learning Settings
```json
{
  "learning": {
    "enabled": true,
    "model_path": "microsoft/DialoGPT-medium",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 0.0001,
    "batch_size": 4,
    "max_length": 2048,
    "training_interval_hours": 6,
    "min_data_points": 10,
    "importance_threshold": 0.5
  }
}
```

#### MCP Settings
```json
{
  "mcp": {
    "enabled": true,
    "timeout": 30,
    "servers": [
      {
        "name": "web-search",
        "command": ["node", "web-search-server.js"]
      },
      {
        "name": "code-tools",
        "url": "http://localhost:8002"
      }
    ]
  }
}
```

## 🔧 LM Studio Integration

### Setup for LM Studio Users

1. **Start LM Studio Server**
   ```bash
   # In LM Studio:
   # 1. Load your model (Llama 3.2 7B recommended)
   # 2. Go to Local Server tab
   # 3. Start server on port 1234
   # 4. Enable CORS if needed
   ```

2. **Update Evo Configuration**
   ```bash
   # Option 1: Use init command
   evo init --model-url http://127.0.0.1:1234 --model-name llama3.2

   # Option 2: Edit config manually
   nano ~/.evo/config.json
   ```

3. **Test Connection**
   ```bash
   # Test the connection
   curl http://127.0.0.1:1234/v1/models

   # Test with Evo
   evo chat
   > "Hello, can you hear me?"
   ```

### LM Studio Configuration File
Create `~/.evo/config.json` specifically for LM Studio:

```json
{
  "model_name": "llama3.2",
  "model_url": "http://127.0.0.1:1234",
  "model_type": "openai_compatible",
  "api_key": "lm-studio",
  "temperature": 0.7,
  "max_tokens": 4096,
  "top_p": 0.9,
  "top_k": 50,
  "use_memory": true,
  "enable_learning": true,
  "enable_mcp": false,
  "database_path": "~/.evo/memory.db",
  "sessions_dir": "~/.evo/sessions",
  "system_prompt": "You are Evo, an AI assistant that learns and remembers. You can access your memory to recall past conversations and learned facts.",
  "memory": {
    "embedding_model": "all-MiniLM-L6-v2",
    "max_episodic_memories": 10000,
    "consolidation_interval_hours": 12,
    "importance_threshold": 0.3
  },
  "learning": {
    "enabled": true,
    "lora_rank": 16,
    "learning_rate": 0.0001,
    "training_interval_hours": 6,
    "min_data_points": 10,
    "importance_threshold": 0.5,
    "adapter_save_path": "~/.evo/adapters"
  },
  "cli": {
    "theme": "dark",
    "auto_save_sessions": true,
    "session_timeout": 1800
  }
}
```

## 🗂️ Directory Structure

After installation, your Evo AI directory structure will be:

```
~/.evo/
├── config.json              # Main configuration
├── memory.db                # SQLite database
├── memory.db-wal            # Write-ahead log
├── memory.db-shm            # Shared memory
├── sessions/                # Saved conversations
│   ├── session_123.json
│   └── session_456.json
├── adapters/                # Learned models
│   ├── adapter_1699123456/
│   └── adapter_1699234567/
├── logs/                    # System logs
│   └── evo.log
└── cache/                   # Temporary files
    └── embeddings/
```

## 🧪 Testing Your Setup

### Basic Functionality Test
```bash
# Test CLI
evo --help

# Test configuration
evo status

# Test memory system
evo teach "My name is Alice"
evo recall "name"

# Test conversation
evo chat
> "Hello, what's my name?"
# Should respond with "Alice"
```

### Learning Test
```bash
# Run the full demo
python scripts/run_demo.py

# This will:
# 1. Test basic memory functions
# 2. Test learning and recall
# 3. Test persistence across sessions
```

### Performance Test
```bash
# Memory performance test
evo memory stats

# Check system resources
evo monitor

# Test with large conversation
evo chat
> "Tell me a long story about artificial intelligence"
# Should handle gracefully
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Model not found" Error
```bash
# Check your model server
curl http://127.0.0.1:1234/v1/models

# Verify config
evo status

# Update model name
evo config set model_name "your-model-name"
```

#### 2. Memory Issues
```bash
# Check database
ls -la ~/.evo/memory.db

# Test memory system
evo memory stats

# Reset if corrupted
rm ~/.evo/memory.db
evo init
```

#### 3. Learning Not Working
```bash
# Check learning status
evo status --learning

# Check logs
tail -f ~/.evo/logs/evo.log

# Disable if problematic
evo config set enable_learning false
```

#### 4. High Resource Usage
```bash
# Monitor usage
evo monitor

# Reduce batch size
evo config set learning.batch_size 2

# Reduce memory limit
evo config set memory.max_episodic_memories 5000
```

### Debug Mode
```bash
# Enable debug logging
export EVO_DEBUG=1
evo chat

# Check logs
tail -f ~/.evo/logs/evo.log
```

### Reset Configuration
```bash
# Backup current config
cp ~/.evo/config.json ~/.evo/config.json.backup

# Reset to defaults
rm ~/.evo/config.json
evo init

# Or specific reset
evo config reset
```

## 🚀 Advanced Setup

### GPU Acceleration
```bash
# Install CUDA dependencies (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Enable GPU in config
evo config set learning.use_gpu true
```

### Custom Models
```json
{
  "model_name": "your-custom-model",
  "model_url": "http://localhost:8080",
  "model_type": "custom",
  "headers": {
    "Authorization": "Bearer your-token"
  }
}
```

### Production Deployment
```bash
# Create production config
cp ~/.evo/config.json ~/.evo/config.prod.json

# Adjust for production
evo config set --config ~/.evo/config.prod.json enable_learning false
evo config set --config ~/.evo/config.prod.json log_level INFO

# Run with production config
EVO_CONFIG=~/.evo/config.prod.json evo chat
```

## 📚 Next Steps

After setup:
1. **Read the [Architecture Documentation](architecture.md)**
2. **Try the [Memory System Guide](memory-system.md)**
3. **Learn about [Memory Consolidation](memory-consolidation.md)**
4. **Explore [MCP Integration](mcp-integration.md)**
5. **Check out [Performance Tuning](performance-tuning.md)**

## 🤝 Getting Help

- **Documentation**: Check other docs in this folder
- **Issues**: Create a GitHub issue
- **Discussions**: Join our community discussions
- **Logs**: Always check `~/.evo/logs/evo.log` for errors

Your Evo AI is now ready to learn and grow with you! 🎉