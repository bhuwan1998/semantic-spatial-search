#!/usr/bin/env bash
#
# setup.sh - Bootstrap the Geo-Agentic Spatial Search project.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script will:
#   1. Check for required system dependencies (Python 3, Docker, Ollama)
#   2. Create a Python virtual environment
#   3. Install Python dependencies
#   4. Start the PostgreSQL container (Docker Compose)
#   5. Pull the required Ollama models (SQL LLM + embedding model)
#   6. Download Adelaide OSM data into PostgreSQL (via data/setup_db.py)
#   7. Copy .env.example to .env if .env doesn't exist

set -euo pipefail

# ---- Helpers ----------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Pre-flight checks -----------------------------------------------------

info "Checking system dependencies..."

# Python 3
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python 3 is required but not found. Install it from https://www.python.org"
fi

PY_VERSION=$($PYTHON --version 2>&1)
info "Found $PY_VERSION"

# Verify minimum Python version (3.11+)
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    error "Python 3.11+ is required (found $PY_VERSION). Please upgrade."
fi

# Docker
if ! command -v docker &>/dev/null; then
    error "Docker is required but not found. Install it from https://www.docker.com/get-started"
fi
info "Found Docker at $(command -v docker)"

# Docker Compose (v2 plugin or standalone)
if docker compose version &>/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
else
    error "Docker Compose is required but not found. Install Docker Desktop or the Compose plugin."
fi
info "Found Docker Compose: $COMPOSE"

# Ollama
if ! command -v ollama &>/dev/null; then
    error "Ollama is required but not found. Install it from https://ollama.com/download"
fi
info "Found ollama at $(command -v ollama)"

# ---- Environment file -------------------------------------------------------

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        info "Created .env from .env.example"
    fi
else
    info ".env already exists, skipping."
fi

# Load env vars for use in this script
# shellcheck disable=SC1091
[ -f ".env" ] && set -o allexport && source .env && set +o allexport || true

# ---- Virtual environment ---------------------------------------------------

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtual environment ($VENV_DIR)"

# ---- Python dependencies ---------------------------------------------------

info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Python dependencies installed."

# ---- Docker: build image and start PostgreSQL -------------------------------

info "Building Docker image (PostgreSQL 15 + PostGIS + pgvector + Apache AGE)..."
info "This may take 5-10 minutes on first run (compiling pgvector from source)."
$COMPOSE build

info "Starting PostgreSQL container..."
$COMPOSE up -d

info "Waiting for PostgreSQL to be ready..."
MAX_WAIT=120
WAITED=0
until $COMPOSE exec db pg_isready -U "${DB_USER:-geo}" -d "${DB_NAME:-geospatial}" &>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        error "PostgreSQL did not become ready within ${MAX_WAIT}s. Check: $COMPOSE logs db"
    fi
done
info "PostgreSQL is ready."

# ---- Ollama models ---------------------------------------------------------

SQL_MODEL="${OLLAMA_MODEL:-gemma4:e4b}"
EMBED_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

info "Pulling Ollama SQL model: $SQL_MODEL..."
if ollama pull "$SQL_MODEL"; then
    info "Model $SQL_MODEL is ready."
else
    warn "Failed to pull $SQL_MODEL. Make sure Ollama is running: ollama serve"
fi

info "Pulling Ollama embedding model: $EMBED_MODEL..."
if ollama pull "$EMBED_MODEL"; then
    info "Model $EMBED_MODEL is ready."
else
    warn "Failed to pull $EMBED_MODEL. Schema RAG will fall back to full schema injection."
fi

# ---- OSM data download and database setup ----------------------------------

info "Setting up Adelaide OSM data (download + PostGIS load + embeddings + AGE graph)..."
info "This will take several minutes on first run."
$PYTHON data/setup_db.py

info "Database setup complete."

# ---- Done ------------------------------------------------------------------

echo ""
echo "=========================================="
info "Setup complete!"
echo ""
echo "  To start the app:"
echo ""
echo "    source .venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "  Make sure Ollama is running:"
echo ""
echo "    ollama serve"
echo ""
echo "  To restart the database container:"
echo ""
echo "    $COMPOSE up -d"
echo ""
echo "  To stop the database:"
echo ""
echo "    $COMPOSE down"
echo ""
echo "=========================================="
