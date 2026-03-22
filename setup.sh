#!/usr/bin/env bash
#
# setup.sh - Bootstrap the Natural Language Spatial Search project.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script will:
#   1. Check for required system dependencies (Python 3, Ollama, libspatialite)
#   2. Create a Python virtual environment
#   3. Install Python dependencies
#   4. Pull the Ollama model (llama3.1:8b)
#   5. Download Adelaide OSM data into a GeoPackage
#   6. Copy .env.example to .env if .env doesn't exist

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
    error "Python 3.11+ is required (found $PY_VERSION). Please upgrade your Python installation."
fi

# Ollama
if ! command -v ollama &>/dev/null; then
    error "Ollama is required but not found. Install it from https://ollama.com/download"
fi
info "Found ollama at $(command -v ollama)"

# libspatialite
SPATIALITE_FOUND=false
if [ "$(uname)" = "Darwin" ]; then
    # macOS: check Homebrew paths
    for candidate in /opt/homebrew/lib/mod_spatialite.dylib /usr/local/lib/mod_spatialite.dylib; do
        if [ -f "$candidate" ]; then
            SPATIALITE_FOUND=true
            info "Found libspatialite at $candidate"
            break
        fi
    done
else
    # Linux: check common paths (x86_64 and aarch64)
    for candidate in /usr/lib/x86_64-linux-gnu/mod_spatialite.so /usr/lib/aarch64-linux-gnu/mod_spatialite.so /usr/lib/mod_spatialite.so /usr/lib64/mod_spatialite.so; do
        if [ -f "$candidate" ]; then
            SPATIALITE_FOUND=true
            info "Found libspatialite at $candidate"
            break
        fi
    done
fi

if [ "$SPATIALITE_FOUND" = false ]; then
    warn "libspatialite not found in standard locations."
    echo ""
    echo "Install it with:"
    if [ "$(uname)" = "Darwin" ]; then
        echo "  brew install spatialite-tools libspatialite"
    else
        echo "  sudo apt-get install -y libsqlite3-mod-spatialite spatialite-bin python3-venv"
    fi
    echo ""
    read -rp "Continue anyway? [y/N] " yn
    case "$yn" in
        [Yy]*) warn "Continuing without confirmed libspatialite..." ;;
        *)     exit 1 ;;
    esac
fi

# ---- Virtual environment ----------------------------------------------------

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtual environment ($VENV_DIR)"

# ---- Python dependencies ---------------------------------------------------

info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Python dependencies installed."

# ---- Ollama model -----------------------------------------------------------

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

info "Pulling Ollama model: $MODEL (this may take a few minutes on first run)..."
if ollama pull "$MODEL"; then
    info "Model $MODEL is ready."
else
    warn "Failed to pull model $MODEL. Make sure the Ollama server is running (ollama serve)."
fi

# ---- Data download ----------------------------------------------------------

GPKG_PATH="data/adelaide_osm.gpkg"

# Ensure data/ directory exists
mkdir -p data

if [ -f "$GPKG_PATH" ]; then
    info "GeoPackage already exists at $GPKG_PATH ($(du -h "$GPKG_PATH" | cut -f1) )"
    read -rp "Re-download data? [y/N] " yn
    case "$yn" in
        [Yy]*) info "Re-downloading..."; $PYTHON data/setup_data.py ;;
        *)     info "Skipping data download." ;;
    esac
else
    info "Downloading Adelaide OSM data (this may take a few minutes)..."
    $PYTHON data/setup_data.py
    info "GeoPackage built at $GPKG_PATH"
fi

# ---- Environment file -------------------------------------------------------

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        info "Created .env from .env.example"
    fi
else
    info ".env already exists, skipping."
fi

# ---- Done -------------------------------------------------------------------

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
echo "=========================================="
