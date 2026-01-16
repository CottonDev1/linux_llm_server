#!/bin/bash
#==============================================================================
# Claude Code Installation Script for Ubuntu Server
#==============================================================================
# This script installs Claude Code CLI tool on Ubuntu Server
#
# Usage: ./install_claude_code.sh
#==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_warning "Running as root. Claude Code will be installed globally."
fi

# Check Node.js version
log_info "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        log_success "Node.js $(node -v) is installed and meets requirements (>=18)"
    else
        log_warning "Node.js $(node -v) is installed but version 18+ is required"
        log_info "Installing Node.js 20.x..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
else
    log_info "Node.js not found. Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Verify Node.js installation
log_info "Node.js version: $(node -v)"
log_info "npm version: $(npm -v)"

# Install Claude Code
log_info "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Verify installation
if command -v claude &> /dev/null; then
    log_success "Claude Code installed successfully!"
    log_info "Version: $(claude --version 2>/dev/null || echo 'unknown')"
else
    log_error "Claude Code installation failed"
    exit 1
fi

# Create .claude directory if it doesn't exist
CLAUDE_DIR="$HOME/.claude"
if [ ! -d "$CLAUDE_DIR" ]; then
    log_info "Creating $CLAUDE_DIR directory..."
    mkdir -p "$CLAUDE_DIR"
fi

# Setup instructions
echo ""
echo "=============================================="
echo "Claude Code Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Run 'claude' to start Claude Code"
echo "2. You'll be prompted to authenticate with your Anthropic API key"
echo ""
echo "For headless server usage, run in tmux or screen:"
echo "  tmux new -s claude"
echo "  claude"
echo "  # Press Ctrl+b d to detach"
echo "  # Run 'tmux attach -t claude' to reconnect"
echo ""
