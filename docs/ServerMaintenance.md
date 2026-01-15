# Ubuntu LLM Server Maintenance Guide

Comprehensive guide for managing the Ubuntu LLM server (10.101.20.21).

---

## Server Overview

| Component | Details |
|-----------|---------|
| **IP Address** | 10.101.20.21 |
| **Username** | chad |
| **GPUs** | GPU 0: RTX 3050 (8GB), GPU 1: GTX 1660 (6GB) |
| **OS** | Ubuntu Server |
| **Key Paths** | `/data/projects/`, `/data/models/`, `/data/logs/` |

---

## Quick Reference

### Service Ports

| Service | Port | Model | GPU |
|---------|------|-------|-----|
| llama-sql | 8080 | SQLCoder-7B | GPU 0 |
| llama-general | 8081 | Qwen2.5-3B | GPU 1 |
| llama-code | 8082 | Qwen2.5-Coder-1.5B | GPU 1 |
| llama-embedding | 8083 | Nomic-Embed | GPU 1 |

### Important Paths

| Path | Description |
|------|-------------|
| `/data/projects/llm_website/` | Main project directory |
| `/data/projects/llama.cpp/` | llama.cpp installation |
| `/data/models/` | Shared model files |
| `/data/logs/llama/` | LLM service logs |
| `/etc/systemd/system/` | Service unit files |

---

## System Monitoring

### GPU Status

```bash
# Check GPU usage, memory, temperature
nvidia-smi

# Continuous monitoring (updates every 2 seconds)
watch -n 2 nvidia-smi

# Detailed GPU info
nvidia-smi -q

# Show only memory usage
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

### System Resources

```bash
# Memory usage
free -h

# Disk space
df -h

# Disk usage by directory
du -sh /data/*
du -sh /data/models/*

# CPU and memory per process
htop
# or
top

# System uptime and load
uptime

# Detailed system info
neofetch
# or
cat /etc/os-release
```

### Network

```bash
# Check listening ports
sudo ss -tlnp

# Check specific port
sudo ss -tlnp | grep 8080

# Check network connections
netstat -an | grep ESTABLISHED

# Test if a service is reachable
curl -s http://localhost:8080/health
curl -s http://localhost:8081/health
curl -s http://localhost:8082/health
curl -s http://localhost:8083/health
```

---

## LLM Service Management (systemd)

### View Service Status

```bash
# Check all LLM services
systemctl status llama-sql llama-general llama-code llama-embedding

# Check single service
systemctl status llama-sql

# List all llama services
systemctl list-units --type=service | grep llama

# Check if services are enabled (start on boot)
systemctl is-enabled llama-sql llama-general llama-code llama-embedding
```

### Start/Stop/Restart Services

```bash
# Start a service
sudo systemctl start llama-sql

# Stop a service
sudo systemctl stop llama-sql

# Restart a service
sudo systemctl restart llama-sql

# Restart all LLM services
sudo systemctl restart llama-sql llama-general llama-code llama-embedding

# Reload service config without restart (if supported)
sudo systemctl reload llama-sql
```

### Enable/Disable Services

```bash
# Enable service to start on boot
sudo systemctl enable llama-sql

# Disable service from starting on boot
sudo systemctl disable llama-sql

# Enable all LLM services
sudo systemctl enable llama-sql llama-general llama-code llama-embedding
```

### View Service Logs

```bash
# View logs for a service
journalctl -u llama-sql

# Follow logs in real-time
journalctl -u llama-sql -f

# View last 100 lines
journalctl -u llama-sql -n 100

# View logs since boot
journalctl -u llama-sql -b

# View logs from last hour
journalctl -u llama-sql --since "1 hour ago"

# View logs from specific time
journalctl -u llama-sql --since "2024-01-15 00:00:00"

# View all LLM logs
journalctl -u 'llama-*' -f
```

### Service Configuration

```bash
# View service unit file
cat /etc/systemd/system/llama-sql.service

# Edit service unit file
sudo nano /etc/systemd/system/llama-sql.service

# After editing, reload systemd
sudo systemctl daemon-reload

# Then restart the service
sudo systemctl restart llama-sql
```

---

## LLM Health Checks & Testing

### Health Endpoints

```bash
# Check all services are responding
for port in 8080 8081 8082 8083; do
    echo -n "Port $port: "
    curl -s http://localhost:$port/health && echo " OK" || echo " FAILED"
done
```

### Test Inference

```bash
# Test SQL model (port 8080)
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "SELECT", "n_predict": 50}'

# Test general model (port 8081)
curl -X POST http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "n_predict": 50}'

# Test code model (port 8082)
curl -X POST http://localhost:8082/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def fibonacci(n):", "n_predict": 100}'

# Test embedding model (port 8083)
curl -X POST http://localhost:8083/embedding \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a test sentence."}'
```

### Check Model Info

```bash
# Get model properties
curl -s http://localhost:8080/props | jq .
curl -s http://localhost:8081/props | jq .
```

---

## Log Management

### View Logs

```bash
# LLM service logs (if using file logging)
tail -f /data/logs/llama/sql.log
tail -f /data/logs/llama/general.log
tail -f /data/logs/llama/code.log
tail -f /data/logs/llama/embedding.log

# View last 100 lines
tail -n 100 /data/logs/llama/sql.log

# Search logs for errors
grep -i error /data/logs/llama/*.log
grep -i "failed\|error\|exception" /data/logs/llama/sql.log

# View logs with timestamps
cat /data/logs/llama/sql.log | less
```

### Log Rotation

```bash
# Check log sizes
ls -lh /data/logs/llama/

# Manual log rotation (truncate)
sudo truncate -s 0 /data/logs/llama/sql.log

# Clear all LLM logs
sudo truncate -s 0 /data/logs/llama/*.log
```

### System Logs

```bash
# General system log
sudo tail -f /var/log/syslog

# Authentication log
sudo tail -f /var/log/auth.log

# Kernel messages
dmesg | tail -50
dmesg -w  # Follow kernel messages
```

---

## Docker Management

---

## Nginx Reverse Proxy

---

## Cockpit Web Management

Cockpit provides a web-based management interface for the server.

### Access

| Setting | Value |
|---------|-------|
| URL | https://10.101.20.21:9090 |
| Login | Use your Linux credentials (chad) |
| Protocol | HTTPS (self-signed certificate) |

### Features

- System overview (CPU, memory, disk)
- Service management (start/stop/restart)
- Log viewer (journalctl)
- Network configuration
- Storage management
- Terminal access
- User management

### Service Management

```bash
# Check status
sudo systemctl status cockpit.socket

# Start/stop
sudo systemctl start cockpit.socket
sudo systemctl stop cockpit.socket

# Enable/disable on boot
sudo systemctl enable cockpit.socket
sudo systemctl disable cockpit.socket

# Restart
sudo systemctl restart cockpit.socket
```

### Troubleshooting

```bash
# Check if listening
sudo ss -tlnp | grep 9090

# Check logs
journalctl -u cockpit -n 50

# Test locally
curl -sk https://localhost:9090/
```

Nginx serves as a reverse proxy, forwarding HTTP requests on port 80 to the Node.js application on port 3000.

### Configuration

| Setting | Value |
|---------|-------|
| Listen port | 80 |
| Proxy target | 127.0.0.1:3000 |
| Config file | /etc/nginx/sites-available/llm-website |
| Access log | /var/log/nginx/llm-website.access.log |
| Error log | /var/log/nginx/llm-website.error.log |

### Service Management

```bash
# Check status
sudo systemctl status nginx

# Start/stop/restart
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx

# Reload config without dropping connections
sudo systemctl reload nginx

# Test configuration before applying
sudo nginx -t
```

### View Logs

```bash
# Access log (requests)
tail -f /var/log/nginx/llm-website.access.log

# Error log
tail -f /var/log/nginx/llm-website.error.log

# Last 100 lines
tail -n 100 /var/log/nginx/llm-website.access.log
```

### Edit Configuration

```bash
# Edit the site config
sudo nano /etc/nginx/sites-available/llm-website

# Test config after editing
sudo nginx -t

# Reload to apply changes
sudo systemctl reload nginx
```

### Troubleshooting

```bash
# Check if Nginx is running
sudo systemctl status nginx

# Check if port 80 is listening
sudo ss -tlnp | grep :80

# Test proxy connection
curl -I http://localhost:80

# Check error log for issues
tail -50 /var/log/nginx/llm-website.error.log

# Verify Node.js backend is running
curl -I http://localhost:3000
```

### Container Operations

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Start/stop/restart container
docker start <container_name>
docker stop <container_name>
docker restart <container_name>

# View container logs
docker logs <container_name>
docker logs -f <container_name>  # Follow
docker logs --tail 100 <container_name>  # Last 100 lines

# Execute command in container
docker exec -it <container_name> bash
docker exec <container_name> <command>
```

### Image Management

```bash
# List images
docker images

# Pull an image
docker pull <image_name>

# Remove unused images
docker image prune

# Remove all unused data
docker system prune -a
```

### Docker Resources

```bash
# Check Docker disk usage
docker system df

# Check container resource usage
docker stats

# Inspect container
docker inspect <container_name>
```

---

## Process Management

### Find Processes

```bash
# Find process by name
ps aux | grep llama
ps aux | grep python
ps aux | grep node

# Find process by port
sudo lsof -i :8080
sudo ss -tlnp | grep 8080

# Process tree
pstree -p
```

### Kill Processes

```bash
# Kill by PID
kill <PID>
kill -9 <PID>  # Force kill

# Kill by name
pkill llama-server
pkill -f "llama-server.*8080"

# Kill all matching
killall llama-server
```

### Background Processes

```bash
# Run in background
./script.sh &

# Run immune to hangup
nohup ./script.sh &

# Disown a running job
disown

# List background jobs
jobs

# Bring to foreground
fg %1
```

---

## File Operations

### Navigation & Listing

```bash
# List with details
ls -la
ls -lah  # Human-readable sizes

# Tree view (if installed)
tree -L 2 /data/projects/

# Find files
find /data -name "*.gguf"
find /data -name "*.log" -mtime -1  # Modified in last day
find /data -size +1G  # Files larger than 1GB
```

### File Content

```bash
# View file
cat filename
less filename  # Paginated
head -n 50 filename  # First 50 lines
tail -n 50 filename  # Last 50 lines

# Follow file changes
tail -f filename

# Search in files
grep "pattern" filename
grep -r "pattern" /data/projects/  # Recursive
grep -i "error" *.log  # Case insensitive
```

### File Management

```bash
# Copy
cp source dest
cp -r source_dir dest_dir  # Recursive

# Move/rename
mv source dest

# Delete
rm filename
rm -rf directory  # Recursive force (CAREFUL!)

# Create directory
mkdir -p /path/to/new/dir

# Change permissions
chmod +x script.sh
chmod 755 directory
chown chad:chad file
```

---

## Network Utilities

### Connectivity

```bash
# Ping
ping -c 4 google.com

# Trace route
traceroute google.com

# DNS lookup
nslookup domain.com
dig domain.com

# Check open ports on remote
nc -zv hostname 8080
```

### Downloads

```bash
# Download file
wget https://example.com/file.zip
curl -O https://example.com/file.zip

# Download with progress
wget --progress=bar https://example.com/large-file.zip

# Resume download
wget -c https://example.com/large-file.zip
```

### Transfer Files

```bash
# SCP to/from server
scp localfile.txt chad@10.101.20.21:/data/projects/
scp chad@10.101.20.21:/data/logs/file.log ./

# Rsync (better for large transfers)
rsync -avz --progress localdir/ chad@10.101.20.21:/data/projects/
```

---

## User & Permissions

---

## Firewall Management (UFW)

The server uses UFW (Uncomplicated Firewall) to control network access. Currently configured to only allow connections from the 10.101.20.0/24 subnet.

### Current Configuration

| Setting | Value |
|---------|-------|
| Default incoming | DENY |
| Default outgoing | ALLOW |
| Allowed subnet | 10.101.20.0/24 |
| SSH access | Port 22 from subnet |

### Check Firewall Status

```bash
# View status and rules
sudo ufw status verbose

# List rules with numbers (for deletion)
sudo ufw status numbered

# Check if firewall is enabled
sudo ufw status | head -1
```

### Manage Rules

```bash
# Allow a specific port from the subnet
sudo ufw allow from 10.101.20.0/24 to any port 8080

# Allow a specific IP address (any port)
sudo ufw allow from 192.168.1.100

# Allow specific IP to specific port
sudo ufw allow from 192.168.1.100 to any port 22

# Deny a specific IP
sudo ufw deny from 192.168.1.100

# Delete a rule by number
sudo ufw status numbered
sudo ufw delete 3

# Delete a rule by specification
sudo ufw delete allow from 192.168.1.100
```

### Enable/Disable Firewall

```bash
# Enable firewall (WARNING: ensure SSH rule exists first!)
sudo ufw enable

# Disable firewall
sudo ufw disable

# Reset all rules to defaults
sudo ufw reset
```

### Adding New Services

When adding a new service that needs external access:

```bash
# 1. Check what ports you need
sudo ss -tlnp | grep LISTEN

# 2. Add firewall rule for the port (subnet only)
sudo ufw allow from 10.101.20.0/24 to any port NEW_PORT

# 3. Verify the rule was added
sudo ufw status numbered
```

### Allowing External Access (Outside Subnet)

If you need to allow access from outside the 10.101.20.0/24 subnet:

```bash
# Allow specific external IP
sudo ufw allow from EXTERNAL_IP

# Allow specific external IP to specific port only
sudo ufw allow from EXTERNAL_IP to any port 8080

# Allow from any IP (CAUTION - opens to internet)
sudo ufw allow 8080
```

### Logging

```bash
# Enable logging
sudo ufw logging on

# Set logging level (low, medium, high, full)
sudo ufw logging medium

# View firewall logs
sudo tail -f /var/log/ufw.log
grep UFW /var/log/syslog | tail -50
```

### Troubleshooting Firewall Issues

```bash
# If locked out, use console access to:
sudo ufw disable

# If a service isn't accessible, check:
# 1. Is the service running?
sudo ss -tlnp | grep PORT

# 2. Is the firewall blocking it?
sudo ufw status verbose

# 3. Add the rule if needed
sudo ufw allow from 10.101.20.0/24 to any port PORT
```

### Quick Reference

| Task | Command |
|------|---------|
| View rules | `sudo ufw status verbose` |
| List numbered | `sudo ufw status numbered` |
| Allow port from subnet | `sudo ufw allow from 10.101.20.0/24 to any port PORT` |
| Allow specific IP | `sudo ufw allow from IP_ADDRESS` |
| Delete rule | `sudo ufw delete NUMBER` |
| Enable | `sudo ufw enable` |
| Disable | `sudo ufw disable` |

### User Info

```bash
# Current user
whoami

# User details
id

# Who is logged in
who
w

# Switch user
su - username
sudo -i  # Root shell
```

### Sudo

```bash
# Run as root
sudo <command>

# Edit sudoers
sudo visudo

# Run as another user
sudo -u username <command>
```

---

## Package Management (apt)

### Update System

```bash
# Update package list
sudo apt update

# Upgrade packages
sudo apt upgrade

# Full upgrade (handles dependencies)
sudo apt full-upgrade

# Update and upgrade in one command
sudo apt update && sudo apt upgrade -y
```

### Install/Remove

```bash
# Install package
sudo apt install package_name

# Remove package
sudo apt remove package_name

# Remove with config files
sudo apt purge package_name

# Auto-remove unused dependencies
sudo apt autoremove
```

### Search

```bash
# Search for package
apt search keyword

# Show package info
apt show package_name

# List installed packages
apt list --installed
apt list --installed | grep keyword
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service status for error
systemctl status llama-sql

# Check recent logs
journalctl -u llama-sql -n 50

# Check if port is in use
sudo ss -tlnp | grep 8080

# Kill process on port
sudo kill $(sudo lsof -t -i:8080)

# Verify model file exists
ls -la /data/projects/llm_website/models/llamacpp/
```

### Out of Memory (GPU)

```bash
# Check GPU memory
nvidia-smi

# Restart services to free memory
sudo systemctl restart llama-sql llama-general llama-code llama-embedding

# Check for zombie processes
ps aux | grep -E 'defunct|Z'
```

### Disk Full

```bash
# Check disk usage
df -h

# Find large files
sudo du -sh /* | sort -rh | head -20
sudo find / -type f -size +1G 2>/dev/null

# Clear package cache
sudo apt clean

# Clear journal logs
sudo journalctl --vacuum-size=100M
```

### Network Issues

```bash
# Check if services are listening
sudo ss -tlnp

# Test connectivity to service
curl -v http://localhost:8080/health

# Check firewall
sudo ufw status

# Allow port through firewall
sudo ufw allow 8080
```

---

## Useful One-Liners

```bash
# Restart all LLM services
sudo systemctl restart llama-{sql,general,code,embedding}

# Check all LLM services status
systemctl status llama-{sql,general,code,embedding} --no-pager

# Test all endpoints
for p in 8080 8081 8082 8083; do echo -n "$p: "; curl -s localhost:$p/health && echo OK || echo FAIL; done

# Watch GPU while services restart
watch -n 1 nvidia-smi

# Monitor all LLM logs
journalctl -u 'llama-*' -f

# Check memory usage of llama processes
ps aux | grep llama-server | awk '{sum+=$6} END {print "Total: " sum/1024 " MB"}'

# Quick system health check
echo "=== Disk ===" && df -h / && echo "=== Memory ===" && free -h && echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

---

## Scheduled Tasks (cron)

```bash
# Edit crontab
crontab -e

# List cron jobs
crontab -l

# Example: Restart services daily at 3am
# 0 3 * * * systemctl restart llama-sql llama-general llama-code llama-embedding

# Example: Clear logs weekly
# 0 0 * * 0 truncate -s 0 /data/logs/llama/*.log
```

---

## Emergency Recovery

### Service Recovery

```bash
# Stop all LLM services
sudo systemctl stop llama-sql llama-general llama-code llama-embedding

# Kill any remaining processes
sudo pkill -f llama-server

# Clear GPU memory (may need reboot if stuck)
sudo nvidia-smi --gpu-reset

# Start services one by one
sudo systemctl start llama-sql
sleep 10
sudo systemctl start llama-general
sleep 5
sudo systemctl start llama-code
sleep 5
sudo systemctl start llama-embedding
```

### System Recovery

```bash
# Reboot
sudo reboot

# Shutdown
sudo shutdown -h now

# Schedule reboot
sudo shutdown -r +5  # Reboot in 5 minutes

# Cancel scheduled shutdown
sudo shutdown -c
```

---

*Last updated: January 2026*

---

## Network Configuration (Netplan)

Netplan is Ubuntu's network configuration tool that uses YAML files to define network settings. It provides an abstraction layer that translates configurations into either systemd-networkd (default) or NetworkManager.

### Configuration File

| Setting | Value |
|---------|-------|
| Config file |  |
| Renderer | systemd-networkd (default) |
| Interface | eno1 (DHCP) |

### View Current Configuration

```bash
# View netplan config
sudo cat /etc/netplan/*.yaml

# Check network interfaces
ip addr show
ip link show

# View routing table
ip route

# Check DNS settings
resolvectl status
cat /etc/resolv.conf
```

### Current Server Configuration

```yaml
# /etc/netplan/50-cloud-init.yaml
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: true
```

### Apply Configuration Changes

```bash
# Test configuration (dry run)
sudo netplan try

# Apply configuration
sudo netplan apply

# Generate backend configs without applying
sudo netplan generate

# Debug configuration issues
sudo netplan --debug apply
```

### Common Configuration Examples

#### Static IP Address

```yaml
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: false
      addresses:
        - 10.101.20.21/24
      routes:
        - to: default
          via: 10.101.20.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

#### DHCP with Static DNS

```yaml
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: true
      dhcp4-overrides:
        use-dns: false
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

#### Multiple Interfaces

```yaml
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: true
    eno2:
      dhcp4: false
      addresses:
        - 192.168.1.100/24
```

### NetworkManager Integration

The server uses systemd-networkd as the backend, but NetworkManager runs alongside for connectivity detection (used by PackageKit/Cockpit).

```bash
# Check NetworkManager status
nmcli general status

# Check device management
nmcli device status

# NetworkManager connectivity check config
cat /etc/NetworkManager/conf.d/99-connectivity.conf
```

**Important:** Do not change the netplan renderer to NetworkManager without careful testing, as this can disrupt network connectivity.

### Troubleshooting

```bash
# If network is down after netplan changes, restore default:
sudo tee /etc/netplan/50-cloud-init.yaml << 'YAML'
network:
  version: 2
  ethernets:
    eno1:
      dhcp4: true
YAML
sudo netplan apply

# Check if interface has IP
ip addr show eno1

# Restart networking
sudo systemctl restart systemd-networkd

# Check for netplan syntax errors
sudo netplan --debug generate

# View systemd-networkd logs
journalctl -u systemd-networkd -n 50
```

### Quick Reference

| Task | Command |
|------|---------|
| View config | `sudo cat /etc/netplan/*.yaml` |
| Test changes | `sudo netplan try` |
| Apply changes | `sudo netplan apply` |
| Debug | `sudo netplan --debug apply` |
| Check IP | `ip addr show eno1` |
| Check routes | `ip route` |
| Check DNS | `resolvectl status` |

