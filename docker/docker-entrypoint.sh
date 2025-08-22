#!/bin/bash
set -e

# Docker entrypoint script for semantic search system

# Default values
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_INDEX_PATH="/app/indices"
DEFAULT_MAX_FEATURES="10000"

# Set environment variables with defaults
export LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}
export INDEX_PATH=${INDEX_PATH:-$DEFAULT_INDEX_PATH}
export MAX_FEATURES=${MAX_FEATURES:-$DEFAULT_MAX_FEATURES}
export PYTHONPATH=/app/src:$PYTHONPATH

# Ensure directories exist
mkdir -p "$INDEX_PATH" /app/logs /app/data

# Function to run health check
health_check() {
    echo "Running health check..."
    python -c "
import sys
try:
    import semantic_search
    from semantic_search import SemanticSearchService
    print('✓ Import successful')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ Health check failed: {e}')
    sys.exit(1)
"
}

# Function to generate sample data
generate_sample_data() {
    echo "Generating sample data..."
    cd /app
    python examples/sample_data/generate_sample_data.py
}

# Function to run basic demo
run_basic_demo() {
    echo "Running basic usage demo..."
    cd /app
    python examples/basic_usage.py
}

# Function to run advanced demo
run_advanced_demo() {
    echo "Running advanced usage demo..."
    cd /app
    python examples/advanced_usage.py
}

# Function to run tests
run_tests() {
    echo "Running test suite..."
    cd /app
    python -m pytest tests/ -v --cov=semantic_search --cov-report=term-missing
}

# Main command handling
case "$1" in
    "health")
        health_check
        ;;
    "generate-data")
        generate_sample_data
        ;;
    "demo")
        run_basic_demo
        ;;
    "advanced-demo")
        run_advanced_demo
        ;;
    "test")
        run_tests
        ;;
    "bash"|"sh")
        exec /bin/bash
        ;;
    "")
        # Default: run basic demo
        echo "Starting semantic search system..."
        echo "Log level: $LOG_LEVEL"
        echo "Index path: $INDEX_PATH"
        echo "Max features: $MAX_FEATURES"
        
        # Run health check first
        health_check
        
        # Generate sample data if needed
        if [ ! -f "/app/examples/sample_data/sample_documents.json" ]; then
            generate_sample_data
        fi
        
        # Run basic demo
        run_basic_demo
        ;;
    *)
        # Execute provided command
        echo "Executing command: $@"
        exec "$@"
        ;;
esac
