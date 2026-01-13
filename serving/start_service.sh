#!/bin/bash
# Simple script to start the recommendation service locally
#
# Usage:
#   ./serving/start_service.sh
#   ./serving/start_service.sh --port 8080
#   ./serving/start_service.sh --reload  # Auto-reload on code changes

set -e  # Exit on error

# Default values
PORT=8000
HOST="0.0.0.0"
RELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--host HOST] [--reload]"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please create one with: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set library path for XGBoost on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "/opt/homebrew/opt/libomp/lib" ]; then
        export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
        echo "Set DYLD_LIBRARY_PATH for XGBoost (macOS)"
    fi
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if ranking model exists
if [ ! -f "ranking/models/xgboost_tuned.json" ]; then
    echo "Warning: Ranking model not found at ranking/models/xgboost_tuned.json"
    echo "Run training first: cd ranking/training && python train_ranking_model.py"
    exit 1
fi

# Start service
echo "=========================================="
echo "Starting Recommendation Service"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "API docs: http://$HOST:$PORT/docs"
echo "Health check: http://$HOST:$PORT/health"
echo "=========================================="
echo ""

# Run uvicorn
uvicorn serving.api:app --host "$HOST" --port "$PORT" $RELOAD
