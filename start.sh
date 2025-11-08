#!/bin/bash
echo "🚀 Iniciando servidor FastAPI en Render..."

APP_PATH="main:app"
HOST="0.0.0.0"
PORT=${PORT:-8000}

# Ejecutar uvicorn usando python -m
exec python -m uvicorn $APP_PATH --host $HOST --port $PORT
