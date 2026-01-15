#!/bin/bash

# Install LLaMA systemd services
# Run as root or with sudo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing LLaMA systemd services..."

# Create log directory
mkdir -p /data/logs/llama
chown -R chad:chad /data/logs/llama

# Stop existing services if running
echo "Stopping any existing services..."
systemctl stop llama-sql llama-code llama-embedding 2>/dev/null

# Kill any running llama-server processes
pkill -f llama-server 2>/dev/null
sleep 2

# Copy service files
echo "Installing service files..."
cp "$SCRIPT_DIR/llama-sql.service" /etc/systemd/system/
cp "$SCRIPT_DIR/llama-code.service" /etc/systemd/system/
cp "$SCRIPT_DIR/llama-embedding.service" /etc/systemd/system/

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable services
echo "Enabling services for auto-start on boot..."
systemctl enable llama-sql
systemctl enable llama-code
systemctl enable llama-embedding

# Start services
echo "Starting services..."
systemctl start llama-sql
sleep 5
systemctl start llama-code
sleep 3
systemctl start llama-embedding
sleep 3

# Check status
echo ""
echo "=== Service Status ==="
systemctl status llama-sql --no-pager -l | head -10
echo ""
systemctl status llama-code --no-pager -l | head -10
echo ""
systemctl status llama-embedding --no-pager -l | head -10

echo ""
echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "Installation complete!"
echo ""
echo "Useful commands:"
echo "  systemctl status llama-sql"
echo "  systemctl status llama-code"
echo "  systemctl status llama-embedding"
echo "  journalctl -u llama-sql -f"
echo "  systemctl restart llama-sql"
