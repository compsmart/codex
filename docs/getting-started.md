# Getting Started with Evo AI

Welcome to Evo AI! This guide will help you get up and running with your learning AI assistant.

## 🎯 What is Evo AI?

Evo AI is a cutting-edge AI assistant that:
- **Learns from conversations** and remembers across sessions
- **Adapts to your preferences** through incremental fine-tuning
- **Stores knowledge persistently** using advanced memory systems
- **Integrates with external tools** via the Model Context Protocol (MCP)
- **Runs locally** for privacy and control

## 🚀 Quick Start (5 Minutes)

### 1. Install LM Studio
- Download from [lmstudio.ai](https://lmstudio.ai)
- Install and launch LM Studio
- Download **Llama 3.2 7B** model
- Start the local server (default: http://127.0.0.1:1234)

### 2. Install Evo AI
```bash
git clone https://github.com/your-repo/evo-ai.git
cd evo-ai
pip install -e .
```

### 3. Initialize Configuration
```bash
evo init --model-url http://127.0.0.1:1234 --model-name llama3.2
```

### 4. Start Chatting!
```bash
evo chat
```

## 💬 Your First Conversation

Let's teach Evo something and see it remember:

### Step 1: Teach Evo About Yourself
```
evo chat
> "Hi! My name is Sarah and I'm a software engineer."
Evo: Hello Sarah! Nice to meet you. What kind of software engineering do you do?

> "I work mainly with Python and I prefer concise explanations."
Evo: Got it! I'll keep that in mind - Python focus and concise responses. What would you like to know?

> exit
```

### Step 2: Test Memory in New Session
```bash
evo chat
> "What's my name and what do I prefer?"
Evo: Your name is Sarah, and you prefer concise explanations. You work with Python as a software engineer.
```

🎉 **It remembered!** This is Evo's persistent memory in action.

## 🧠 Key Features Explained

### 1. **Persistent Memory**
Evo remembers facts across sessions:
```bash
# Teach facts
evo teach "My favorite programming language is Rust"

# They persist forever
evo recall "favorite programming language"
# Output: "Your favorite programming language is Rust"
```

### 2. **Learning from Conversations**
Evo improves responses based on your interactions:
- Adapts to your communication style
- Learns your preferences and interests
- Builds domain knowledge from discussions

### 3. **Memory Types**
- **Episodic**: Remembers specific conversations
- **Semantic**: Stores facts and knowledge
- **Procedural**: Learns skills and procedures

### 4. **Context-Aware Responses**
Evo uses past conversations to inform current responses:
```
You: "What should I use for my new project?"
Evo: "Given your preference for Rust and work in systems programming,
      I'd recommend Rust for performance-critical applications..."
```

## 🎯 Common Use Cases

### Personal Assistant
```bash
evo teach "I have a meeting with the design team every Tuesday at 2 PM"
evo recall "meeting schedule"
```

### Learning Companion
```bash
# Evo learns as you learn
> "I'm studying machine learning. What's a neural network?"
> "That's helpful. I'm particularly interested in computer vision."
> "Show me how to implement a simple CNN in PyTorch"
# Future responses will be tailored to your ML learning journey
```

### Coding Assistant
```bash
# Remembers your coding preferences
> "I prefer functional programming style"
> "Help me write a Python function to process user data"
# Will suggest functional approaches in future responses
```

### Knowledge Base
```bash
# Build a personal knowledge base
evo teach "The project deadline is December 15th"
evo teach "Use staging environment for testing"
evo teach "Contact John for database issues"

# Later...
evo recall "project deadline"
evo recall "database"
```

## 🔧 Essential Commands

### Chat Commands
```bash
evo chat                    # Start interactive chat
evo chat --session <id>     # Resume specific session
```

### Memory Commands
```bash
evo teach "fact"           # Teach Evo something
evo recall "query"         # Search memory
evo memory stats           # View memory statistics
```

### System Commands
```bash
evo status                 # Check system status
evo sessions list          # View chat sessions
evo config show           # View configuration
```

### In-Chat Commands
```
/help          # Show help
/status        # Agent status
/memory <query> # Search memory
/teach <info>  # Teach information
/clear         # Clear screen
/exit          # Exit chat
```

## ⚙️ Configuration

### Basic Settings
```bash
# View current config
evo config show

# Update settings
evo config set temperature 0.8
evo config set enable_learning true
```

### For LM Studio Users
Your config should look like this:
```json
{
  "model_name": "llama3.2",
  "model_url": "http://127.0.0.1:1234",
  "temperature": 0.7,
  "use_memory": true,
  "enable_learning": true
}
```

## 🎓 Learning How Evo Learns

### Automatic Learning
Evo learns automatically from your conversations:
- **Every 6 hours** (configurable)
- **After meaningful conversations** (10+ quality exchanges)
- **When you teach explicit facts**

### Manual Learning
```bash
# Force immediate learning
evo learn --force

# Teach specific information
evo teach "I prefer detailed code comments"
```

### Learning Progress
```bash
# Check learning status
evo status --learning

# View learning history
evo learning history
```

## 🔍 Advanced Features

### Session Management
```bash
# List all sessions
evo sessions list

# Search sessions
evo sessions search "machine learning"

# View session summary
evo sessions summary <session-id>
```

### Memory Search
```bash
# Search by content
evo recall "Python programming"

# Search by type
evo memory search --type semantic "preferences"

# View memory stats
evo memory stats
```

### Background Services
```bash
# Check background services
evo monitor

# View service logs
tail -f ~/.evo/logs/evo.log
```

## 🛠️ Troubleshooting

### Common Issues

#### "Connection refused" error
```bash
# Check if LM Studio is running
curl http://127.0.0.1:1234/v1/models

# Restart LM Studio server
# Update config if needed
evo config set model_url "http://127.0.0.1:1234"
```

#### Memory not working
```bash
# Check memory stats
evo memory stats

# Reset memory if corrupted
rm ~/.evo/memory.db
evo init
```

#### High resource usage
```bash
# Monitor resources
evo monitor

# Reduce memory usage
evo config set memory.max_episodic_memories 5000
```

### Getting Help
```bash
# Built-in help
evo --help
evo chat --help

# Check logs
tail -f ~/.evo/logs/evo.log

# Debug mode
EVO_DEBUG=1 evo chat
```

## 📊 Success Test

Run this test to verify everything works:

### Test 1: Basic Memory
```bash
evo teach "The capital of Mars is Olympia City"
evo recall "capital of Mars"
# Should return: "The capital of Mars is Olympia City"
```

### Test 2: Cross-Session Persistence
```bash
# Session 1
evo chat
> "Remember: my lucky number is 42"
> exit

# Session 2
evo chat
> "What's my lucky number?"
# Should respond: "Your lucky number is 42"
```

### Test 3: Learning from Conversation
```bash
evo chat
> "I prefer short, direct answers"
> "Explain quantum computing"
> exit

# Later...
evo chat
> "What is artificial intelligence?"
# Should give a concise response based on your preference
```

## 🚀 Next Steps

Now that you're set up:

1. **Explore the [Architecture](architecture.md)** - Understand how Evo works
2. **Read about [Memory Consolidation](memory-consolidation.md)** - How learning happens
3. **Check out [Learning Process](learning-process.md)** - Deep dive into adaptation
4. **Try [MCP Integration](mcp-integration.md)** - Connect external tools

## 💡 Tips for Best Results

### Teaching Evo Effectively
- **Be explicit**: "Remember that I prefer..."
- **Use natural language**: "My favorite way to..."
- **Provide context**: "When coding in Python, I like to..."

### Building Better Conversations
- **Ask follow-up questions** to build knowledge
- **Correct Evo when needed** - it learns from feedback
- **Be consistent** with preferences and style

### Memory Management
- **Regular use** helps build better models
- **Diverse conversations** improve adaptation
- **Explicit teaching** for important facts

Welcome to the future of AI assistance! Evo is ready to learn and grow with you. 🎉