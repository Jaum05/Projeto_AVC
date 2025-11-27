#!/bin/bash
echo "🌐 Iniciando API..."
uvicorn src.app:app --host 0.0.0.0 --port 8000
