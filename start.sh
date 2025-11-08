#!/bin/bash
echo "🚀 Iniciando servidor FastAPI..."

# Confirmar ruta actual
echo "📂 Directorio actual: $(pwd)"

# Variables de configuración
APP_PATH="main:app"         # Ruta de tu app FastAPI (main.py)
HOST="0.0.0.0"              # Escucha en todas las interfaces
PORT=${PORT:-8000}          # Render asigna automáticamente un puerto si existe la variable PORT

# Activar virtualenv si existe (opcional)
if [ -f "venv/bin/activate" ]; then
    echo "🔹 Activando virtualenv..."
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    echo "🔹 Activando virtualenv Windows..."
    source venv/Scripts/activate
fi

# Ejecutar servidor con uvicorn usando Python
echo "⚡ Ejecutando: python -m uvicorn $APP_PATH --host $HOST --port $PORT --reload"
exec python -m uvicorn $APP_PATH --host $HOST --port $PORT --reload
