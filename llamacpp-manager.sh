#!/bin/zsh

# Llama.cpp Server Manager for Alim AI
# Manages embeddings, chat, and reranker servers

setopt errexit

# Configuration
BASE_DIR="/Users/dasword/Projects/alimai"
MODELS_DIR="$BASE_DIR/models"
LOGS_DIR="$BASE_DIR/logs"
PIDS_DIR="$BASE_DIR/logs/pids"

# Server configurations
typeset -A SERVERS
SERVERS=(
    embeddings 8001
    chat 8002
    reranker 8003
)

typeset -A MODELS
MODELS=(
    embeddings "$MODELS_DIR/embeddinggemma-300m-qat-Q4_K_M.gguf"
    chat "$MODELS_DIR/Qwen3-8B-Q4_K_M.gguf"
    reranker "$MODELS_DIR/Qwen3-Reranker-0.6B-Q4_K_M.gguf"
)

get_command() {
    local service=$1
    case $service in
        embeddings)
            echo "llama-server -m ${MODELS[embeddings]} --port 8001 --host 0.0.0.0 -n 512 --embedding"
            ;;
        chat)
            echo "llama-server -m ${MODELS[chat]} --port 8002 --host 0.0.0.0 -n 2048 -c 4096 --jinja -ngl 99 -sm row --no-context-shift --temp 0.6 --top-k 20 --top-p 0.8 --min-p 0 --presence-penalty 1.5"
            ;;
        reranker)
            echo "llama-server -m ${MODELS[reranker]} --port 8003 --host 0.0.0.0 -n 512"
            ;;
    esac
}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$LOGS_DIR" "$PIDS_DIR"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_pid() {
    local service=$1
    local pid_file="$PIDS_DIR/$service.pid"
    
    if [ -f "$pid_file" ]; then
        cat "$pid_file"
    fi
}

is_running() {
    local service=$1
    local pid=$(get_pid "$service")
    
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

check_port() {
    local port=$1
    lsof -i ":$port" -sTCP:LISTEN -t >/dev/null 2>&1
}

# Start a specific server
start_server() {
    local service=$1
    local port=${SERVERS[$service]}
    local model=${MODELS[$service]}
    local command=$(get_command "$service")
    local pid_file="$PIDS_DIR/$service.pid"
    local log_file="$LOGS_DIR/$service.log"
    
    if [ -z "$port" ]; then
        log_error "Unknown service: $service"
        return 1
    fi
    
    # Check if already running
    if is_running "$service"; then
        log_warning "$service is already running (PID: $(get_pid $service))"
        return 0
    fi
    
    # Check if model file exists
    if [ ! -f "$model" ]; then
        log_error "Model file not found: $model"
        return 1
    fi
    
    # Check if port is already in use
    if check_port "$port"; then
        log_error "Port $port is already in use"
        return 1
    fi
    
    log_info "Starting $service server on port $port..."
    
    # Start the server
    cd "$BASE_DIR"
    eval "nohup $command > '$log_file' 2>&1 &"
    local pid=$!
    echo $pid > "$pid_file"
    
    # Wait a moment and verify it started
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        log_success "$service started (PID: $pid)"
        return 0
    else
        log_error "$service failed to start. Check logs: tail -f $log_file"
        rm -f "$pid_file"
        return 1
    fi
}

# Stop a specific server
stop_server() {
    local service=$1
    local pid=$(get_pid "$service")
    local pid_file="$PIDS_DIR/$service.pid"
    
    if [ -z "$pid" ]; then
        log_warning "$service is not running (no PID file)"
        return 0
    fi
    
    if kill -0 "$pid" 2>/dev/null; then
        log_info "Stopping $service (PID: $pid)..."
        kill "$pid"
        
        # Wait for process to stop
        local timeout=10
        while kill -0 "$pid" 2>/dev/null && [ $timeout -gt 0 ]; do
            sleep 1
            ((timeout--))
        done
        
        if kill -0 "$pid" 2>/dev/null; then
            log_warning "$service didn't stop gracefully, forcing..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        
        rm -f "$pid_file"
        log_success "$service stopped"
    else
        log_warning "$service PID $pid is not running"
        rm -f "$pid_file"
    fi
}

# Restart a server
restart_server() {
    local service=$1
    log_info "Restarting $service..."
    stop_server "$service"
    sleep 1
    start_server "$service"
}

# Get status of a server
get_status() {
    local service=$1
    local port=${SERVERS[$service]}
    local pid=$(get_pid "$service")
    
    if is_running "$service"; then
        # Try to get health status
        local health=$(curl -s "http://localhost:$port/health" 2>/dev/null || echo '{"status":"unknown"}')
        local health_status=$(echo "$health" | jq -r '.status' 2>/dev/null || echo "unknown")
        
        echo -e "${GREEN}●${NC} $service (PID: $pid, Port: $port) - $health_status"
        
        # Try to get model info
        local model_info=$(curl -s "http://localhost:$port/v1/models" 2>/dev/null)
        if [ -n "$model_info" ]; then
            local model_name=$(echo "$model_info" | jq -r '.data[0].id' 2>/dev/null | xargs basename 2>/dev/null)
            local params=$(echo "$model_info" | jq -r '.data[0].meta.n_params' 2>/dev/null)
            if [ -n "$params" ] && [ "$params" != "null" ]; then
                local params_m=$(echo "scale=0; $params / 1000000" | bc)
                echo "  Model: $model_name (${params_m}M params)"
            fi
        fi
    else
        echo -e "${RED}○${NC} $service (Port: $port) - stopped"
    fi
}

# Get detailed stats
get_stats() {
    local service=$1
    local port=${SERVERS[$service]}
    
    if ! is_running "$service"; then
        log_warning "$service is not running"
        return 1
    fi
    
    echo -e "\n${BLUE}=== $service Statistics ===${NC}"
    
    # Get model info
    local model_data=$(curl -s "http://localhost:$port/v1/models" 2>/dev/null | jq -r '.data[0]' 2>/dev/null)
    
    if [ -n "$model_data" ]; then
        echo "Model: $(echo "$model_data" | jq -r '.id' | xargs basename)"
        echo "Parameters: $(echo "$model_data" | jq -r '.meta.n_params' | awk '{printf "%.0fM", $1/1000000}')"
        echo "Context Length: $(echo "$model_data" | jq -r '.meta.n_ctx_train') tokens"
        echo "Embedding Dim: $(echo "$model_data" | jq -r '.meta.n_embd')"
        echo "Vocab Size: $(echo "$model_data" | jq -r '.meta.n_vocab')"
    fi
    
    # Process info
    local pid=$(get_pid "$service")
    if [ -n "$pid" ]; then
        echo ""
        ps -p "$pid" -o pid,ppid,%cpu,%mem,vsz,rss,etime,command | head -2 | tail -1 | \
            awk '{printf "PID: %s\nCPU: %s%%\nMemory: %s%% (%s MB)\nUptime: %s\n", $1, $3, $4, int($6/1024), $7}'
    fi
}

# Show usage
show_usage() {
    local script_name="llama-manager.sh"
    cat << EOF
${BLUE}Llama.cpp Server Manager${NC}

${GREEN}Usage:${NC}
  ./$script_name <command> [service]

${GREEN}Commands:${NC}
  start [service]     Start a server or all servers
  stop [service]      Stop a server or all servers
  restart [service]   Restart a server or all servers
  status              Show status of all servers
  stats [service]     Show detailed statistics for a server
  logs [service]      Show logs for a server
  health              Check health of all running servers
  test [service]      Run a quick test on a server
  stream [prompt]     Test streaming chat response

${GREEN}Services:${NC}
  embeddings          Embedding model server (port 8001)
  chat                Chat LLM server (port 8002)
  reranker            Reranker model server (port 8003)
  all                 All servers (default for start/stop/restart)

${GREEN}Examples:${NC}
  ./$script_name start                    # Start all servers
  ./$script_name start chat               # Start only chat server
  ./$script_name stop embeddings          # Stop embeddings server
  ./$script_name status                   # Show status of all servers
  ./$script_name stats chat               # Show detailed stats for chat server
  ./$script_name logs chat                # Tail logs for chat server
  ./$script_name test embeddings          # Test embeddings server
  ./$script_name stream "What is Islam?"  # Test streaming response

EOF
}

# Test a server
test_server() {
    local service=$1
    local port=${SERVERS[$service]}
    
    if ! is_running "$service"; then
        log_error "$service is not running"
        return 1
    fi
    
    log_info "Testing $service server..."
    
    case $service in
        embeddings)
            local result=$(curl -s "http://localhost:$port/v1/embeddings" \
                -H "Content-Type: application/json" \
                -d '{"input":"test","model":"embeddinggemma"}' 2>/dev/null)
            local dims=$(echo "$result" | jq -r '.data[0].embedding | length' 2>/dev/null)
            if [ "$dims" = "768" ]; then
                log_success "Embeddings server working (768 dimensions)"
            else
                log_error "Embeddings server test failed"
            fi
            ;;
        chat)
            local result=$(curl -s "http://localhost:$port/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Say hi /no_think"}],"max_tokens":5}' 2>/dev/null)
            local response=$(echo "$result" | jq -r '.choices[0].message.content' 2>/dev/null)
            if [ -n "$response" ] && [ "$response" != "null" ]; then
                log_success "Chat server working: $response"
            else
                log_error "Chat server test failed"
            fi
            ;;
        reranker)
            local result=$(curl -s "http://localhost:$port/v1/models" 2>/dev/null)
            local model=$(echo "$result" | jq -r '.data[0].id' 2>/dev/null)
            if [ -n "$model" ]; then
                log_success "Reranker server working"
            else
                log_error "Reranker server test failed"
            fi
            ;;
    esac
}

# Stream chat response
stream_chat() {
    local prompt=${1:-"Tell me about Islam in 2 sentences /no_think"}
    local port=8002
    
    if ! is_running "chat"; then
        log_error "Chat server is not running"
        return 1
    fi
    
    log_info "Streaming response for: \"$prompt\""
    echo ""
    
    # Stream and parse responses
    curl -N http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"qwen3-8b\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"stream\": true,
            \"max_tokens\": 200
        }" 2>/dev/null | while IFS= read -r line; do
            if [[ "$line" == data:* ]]; then
                local json="${line#data: }"
                if [ "$json" != "[DONE]" ]; then
                    local content=$(echo "$json" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
                    if [ -n "$content" ] && [ "$content" != "null" ]; then
                        printf "%s" "$content"
                    fi
                fi
            fi
        done
    
    echo ""
    echo ""
    log_success "Streaming complete"
}

# Check health of all servers
check_health() {
    echo -e "\n${BLUE}=== Health Check ===${NC}\n"
    
    for service in ${(k)SERVERS}; do
        get_status "$service"
    done
    
    echo ""
}

# Show logs
show_logs() {
    local service=$1
    local log_file="$LOGS_DIR/$service.log"
    
    if [ ! -f "$log_file" ]; then
        log_error "Log file not found: $log_file"
        return 1
    fi
    
    log_info "Showing logs for $service (Ctrl+C to exit)..."
    tail -f "$log_file"
}

# Main command handler
case "${1:-}" in
    start)
        service=${2:-all}
        if [ "$service" = "all" ]; then
            for s in embeddings chat reranker; do
                start_server "$s"
            done
        else
            start_server "$service"
        fi
        ;;
    
    stop)
        service=${2:-all}
        if [ "$service" = "all" ]; then
            for s in embeddings chat reranker; do
                stop_server "$s"
            done
        else
            stop_server "$service"
        fi
        ;;
    
    restart)
        service=${2:-all}
        if [ "$service" = "all" ]; then
            for s in embeddings chat reranker; do
                restart_server "$s"
            done
        else
            restart_server "$service"
        fi
        ;;
    
    status)
        check_health
        ;;
    
    stats)
        service=${2:-}
        if [ -z "$service" ]; then
            for s in embeddings chat reranker; do
                if is_running "$s"; then
                    get_stats "$s"
                fi
            done
        else
            get_stats "$service"
        fi
        ;;
    
    logs)
        service=${2:-chat}
        show_logs "$service"
        ;;
    
    health)
        check_health
        ;;
    
    test)
        service=${2:-}
        if [ -z "$service" ]; then
            for s in embeddings chat reranker; do
                if is_running "$s"; then
                    test_server "$s"
                fi
            done
        else
            test_server "$service"
        fi
        ;;
    
    stream)
        prompt=${2:-"Tell me about Islam in 2 sentences /no_think"}
        stream_chat "$prompt"
        ;;
    
    *)
        show_usage
        exit 1
        ;;
esac

