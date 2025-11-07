# Llama.cpp Server Manager

## Quick Start

```bash
# Start all servers (embeddings, chat, reranker)
./llama-manager.sh start

# Check status of all running servers
./llama-manager.sh status

# Test all servers
./llama-manager.sh test
```

## Commands

### Start/Stop/Restart

```bash
# Start all servers
./llama-manager.sh start

# Start a specific server
./llama-manager.sh start chat
./llama-manager.sh start embeddings
./llama-manager.sh start reranker

# Stop all servers
./llama-manager.sh stop

# Stop a specific server
./llama-manager.sh stop chat

# Restart all servers
./llama-manager.sh restart

# Restart a specific server
./llama-manager.sh restart chat
```

### Status & Monitoring

```bash
# Show status of all servers
./llama-manager.sh status

# Show health check
./llama-manager.sh health

# Show detailed stats for a server
./llama-manager.sh stats chat
./llama-manager.sh stats embeddings
./llama-manager.sh stats reranker

# Show stats for all running servers
./llama-manager.sh stats
```

### Testing & Logs

```bash
# Test a specific server
./llamacpp-manager.sh test embeddings
./llamacpp-manager.sh test chat
./llamacpp-manager.sh test reranker

# Test all running servers
./llamacpp-manager.sh test

# Test streaming response (real-time token generation)
./llamacpp-manager.sh stream "What is Islam?"
./llamacpp-manager.sh stream "Explain the five pillars /no_think"

# View logs (tail -f)
./llamacpp-manager.sh logs chat
./llamacpp-manager.sh logs embeddings
./llamacpp-manager.sh logs reranker
```

## Server Information

### Embeddings Server (Port 8001)
- **Model**: embeddinggemma-300m-qat (302M params)
- **Purpose**: Generate text embeddings (768 dimensions)
- **API**: OpenAI-compatible embeddings endpoint
- **Endpoint**: `http://localhost:8001/v1/embeddings`

### Chat Server (Port 8002)
- **Model**: Qwen3-8B (8.2B params)
- **Purpose**: Chat completions with reasoning capability
- **API**: OpenAI-compatible chat endpoint
- **Endpoint**: `http://localhost:8002/v1/chat/completions`
- **Special Features**:
  - Add `/no_think` for fast, direct answers
  - Add `/think` for reasoning mode (math, coding, logic)
  - Uses Jinja templates for proper chat formatting

### Reranker Server (Port 8003)
- **Model**: Qwen3-Reranker-0.6B (595M params)
- **Purpose**: Rerank search results for better relevance
- **API**: OpenAI-compatible endpoint
- **Endpoint**: `http://localhost:8003/v1/models`

## Example API Usage

### Embeddings
```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "What is Islam?", "model": "embeddinggemma"}'
```

### Chat (Non-streaming)
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "What is Islam? /no_think"}
    ],
    "max_tokens": 100
  }'
```

### Chat (Streaming) - Real-time token generation
```bash
# Stream response as tokens are generated
curl -N http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "Explain Islam in detail /no_think"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

### Chat (Thinking mode)
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "Solve: 2x + 5 = 13 /think"}
    ],
    "max_tokens": 200
  }'
```

### Python Example (Streaming)
```python
import httpx

url = "http://localhost:8002/v1/chat/completions"
data = {
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "What is Islam? /no_think"}],
    "stream": True,
    "max_tokens": 200
}

with httpx.stream("POST", url, json=data, timeout=30) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            json_str = line[6:]  # Remove "data: " prefix
            if json_str.strip() == "[DONE]":
                break
            try:
                import json
                chunk = json.loads(json_str)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
            except:
                pass
    print()  # New line at end
```

## Files & Locations

- **Script**: `llama-manager.sh`
- **Models**: `models/` directory
- **Logs**: `logs/embeddings.log`, `logs/chat.log`, `logs/reranker.log`
- **PIDs**: `logs/pids/` directory

## Performance Notes

- Running natively on Apple Silicon (ARM64)
- Chat: ~25 tokens/second
- Embeddings: ~0.2 seconds per request
- All models use Q4_K_M quantization for optimal speed/quality balance
- Total RAM usage: ~6-7 GB for all three servers

## Troubleshooting

### Server won't start
```bash
# Check if port is already in use
lsof -i :8001
lsof -i :8002
lsof -i :8003

# Check logs for errors
./llama-manager.sh logs chat
```

### Check model files
```bash
ls -lh models/*.gguf
```

Expected files:
- `embeddinggemma-300m-qat-Q4_K_M.gguf` (~225 MB)
- `Qwen3-8B-Q4_K_M.gguf` (~4.7 GB)
- `Qwen3-Reranker-0.6B-Q4_K_M.gguf` (~378 MB)

### Kill all servers manually
```bash
pkill -f llama-server
```

## Optional: Add to PATH

For easier access from anywhere:

```bash
# Add to ~/.zshrc
echo 'alias llama-manager="~/Projects/alimai/llama-manager.sh"' >> ~/.zshrc
source ~/.zshrc

# Then use from anywhere:
llama-manager status
llama-manager start
```

## Resources

- [Qwen3-8B Documentation](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Llama.cpp Server Documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Project Setup Guide](LLAMACPP_SETUP.md)

