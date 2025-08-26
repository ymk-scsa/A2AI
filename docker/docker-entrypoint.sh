#!/bin/bash
set -e

echo "Starting A2AI environment..."

if [ "$1" = "jupyter-lab" ]; then
    echo "Starting Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
elif [ "$1" = "api" ]; then
    echo "Starting FastAPI server..."
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
else
    exec "$@"
fi