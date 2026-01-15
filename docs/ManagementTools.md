# Server Management Tools Guide

This document covers the management and monitoring tools available on the Ubuntu LLM Server (10.101.20.21).

---

## Quick Access Reference

| Tool | URL | Default Credentials | Purpose |
|------|-----|---------------------|---------|
| **Cockpit** | https://10.101.20.21:9090 | Linux credentials (chad/admin) | System administration |
| **Grafana** | http://10.101.20.21:3001 | admin/admin | Metrics visualization |
| **Prometheus** | http://10.101.20.21:9091 | None | Metrics collection |
| **Prefect** | http://10.101.20.21:4200 | None | Workflow orchestration |

---

## 1. Cockpit - System Administration

### Purpose
Cockpit provides a web-based interface for managing the server, including:
- Real-time system monitoring (CPU, memory, disk, network)
- Service management (start/stop/restart systemd services)
- Log viewer (journalctl with filtering)
- Network configuration
- Storage management
- User management
- Terminal access (web-based shell)

### Accessing Cockpit
1. Open browser to: **https://10.101.20.21:9090**
2. Accept the self-signed certificate warning
3. Login with Linux credentials:
   - Username: `chad`
   - Password: `admin`

### Key Features
| Tab | Function |
|-----|----------|
| Overview | System health dashboard |
| Logs | Browse and filter system logs |
| Storage | Disk usage, mount points |
| Networking | Network interfaces, firewall |
| Services | Manage systemd services |
| Terminal | Web-based shell access |

### Start/Stop Cockpit
```bash
# Check status
sudo systemctl status cockpit.socket

# Start
sudo systemctl start cockpit.socket

# Stop
sudo systemctl stop cockpit.socket

# Enable on boot
sudo systemctl enable cockpit.socket
```

---

## 2. Grafana - Metrics Visualization

### Purpose
Grafana provides dashboards for visualizing metrics collected by Prometheus:
- LLM service performance (tokens/sec, latency, requests)
- System metrics (CPU, memory, disk I/O, network)
- Historical trends and alerting

### Accessing Grafana
1. Open browser to: **http://10.101.20.21:3001**
2. Login with default credentials:
   - Username: `admin`
   - Password: `admin` (change on first login)

### Pre-configured Data Sources
- **Prometheus**: http://localhost:9091 (default)

### Creating LLM Dashboard
1. Click **+ → Dashboard**
2. Add a new panel
3. Use these example queries for LLM metrics:

| Metric | PromQL Query |
|--------|--------------|
| Tokens/second | `llamacpp:predicted_tokens_seconds` |
| Prompt tokens/sec | `llamacpp:prompt_tokens_seconds` |
| Active requests | `llamacpp:requests_processing` |
| Total tokens generated | `llamacpp:tokens_predicted_total` |

### Start/Stop Grafana
```bash
# Check status
sudo systemctl status grafana-server

# Start
sudo systemctl start grafana-server

# Stop
sudo systemctl stop grafana-server

# Restart
sudo systemctl restart grafana-server

# View logs
sudo journalctl -u grafana-server -f
```

---

## 3. Prometheus - Metrics Collection

### Purpose
Prometheus scrapes and stores time-series metrics from:
- LLM services (llama.cpp on ports 8080-8083)
- Node exporter (system metrics)
- Custom application metrics

### Accessing Prometheus
1. Open browser to: **http://10.101.20.21:9091**
2. No authentication required

### Key Pages
| Page | URL | Purpose |
|------|-----|---------|
| Graph | /graph | Query and visualize metrics |
| Targets | /targets | View scrape targets and health |
| Config | /config | View current configuration |
| Alerts | /alerts | View active alerts |

### Configured Scrape Targets
| Job | Target | Metrics |
|-----|--------|---------|
| llama-sql | localhost:8080 | SQL model performance |
| llama-general | localhost:8081 | General model performance |
| llama-code | localhost:8082 | Code model performance |
| llama-embedding | localhost:8083 | Embedding model performance |
| node | localhost:9100 | System metrics |
| prometheus | localhost:9091 | Prometheus self-metrics |

### Start/Stop Prometheus
```bash
# Check status
sudo systemctl status prometheus

# Start
sudo systemctl start prometheus

# Stop
sudo systemctl stop prometheus

# Restart
sudo systemctl restart prometheus

# Reload config (without restart)
sudo systemctl reload prometheus

# View logs
sudo journalctl -u prometheus -f
```

### Configuration Files
- Main config: `/etc/prometheus/prometheus.yml`
- Arguments: `/etc/default/prometheus`

---

## 4. Prefect - Workflow Orchestration

### Purpose
Prefect orchestrates and schedules data pipelines and workflows:
- Audio processing pipelines
- Document ingestion workflows
- Scheduled batch jobs
- Pipeline monitoring and retries

### Accessing Prefect
1. Open browser to: **http://10.101.20.21:4200**
2. No authentication required by default

### Key Concepts
| Term | Description |
|------|-------------|
| Flow | A Python function decorated with @flow |
| Task | A unit of work within a flow |
| Deployment | A scheduled/triggered flow |
| Work Pool | Execution environment for flows |
| Work Queue | Queue for pending flow runs |

### Start/Stop Prefect Server
```bash
# Start server (foreground)
prefect server start --host 0.0.0.0

# Start server (background)
nohup prefect server start --host 0.0.0.0 > /var/log/prefect/server.log 2>&1 &

# Check if running
curl http://localhost:4200/api/health

# Stop (if running in background)
pkill -f "prefect server"
```

### Using Prefect
```python
from prefect import flow, task

@task
def process_data(data):
    return data.upper()

@flow
def my_pipeline(input_data: str):
    result = process_data(input_data)
    return result

# Run directly
my_pipeline("hello")

# Create deployment for scheduling
# prefect deployment build my_pipeline.py:my_pipeline -n "My Pipeline"
```

### Common Commands
```bash
# List flows
prefect flow ls

# List deployments
prefect deployment ls

# Run a deployment
prefect deployment run "flow-name/deployment-name"

# View logs
prefect flow-run logs <flow-run-id>
```

---

## Firewall Configuration

All tools are accessible from the **10.101.20.0/24** subnet only.

### Current Port Configuration
| Port | Service |
|------|---------|
| 22 | SSH |
| 80 | Nginx (website) |
| 3000 | Node.js (direct) |
| 3001 | Grafana |
| 4200 | Prefect |
| 8080-8083 | LLM Services |
| 9090 | Cockpit |
| 9091 | Prometheus |
| 9100 | Node Exporter |

### Managing Firewall
```bash
# View current rules
sudo ufw status verbose

# Allow new port from subnet
sudo ufw allow from 10.101.20.0/24 to any port <PORT>

# Delete a rule
sudo ufw status numbered
sudo ufw delete <NUMBER>
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check service status
sudo systemctl status <service-name>

# View recent logs
sudo journalctl -u <service-name> -n 50

# Check if port is in use
sudo ss -tlnp | grep <PORT>
```

### Cannot Access Web Interface
1. Verify service is running: `sudo systemctl status <service>`
2. Check firewall: `sudo ufw status`
3. Verify port is listening: `sudo ss -tlnp | grep <PORT>`
4. Test locally: `curl http://localhost:<PORT>/`

### LLM Metrics Not Showing
1. Verify LLM service has --metrics flag:
   ```bash
   cat /etc/systemd/system/llama-general.service | grep metrics
   ```
2. Check metrics endpoint:
   ```bash
   curl http://localhost:8081/metrics
   ```
3. Check Prometheus targets:
   ```bash
   curl http://localhost:9091/api/v1/targets
   ```

---

*Last updated: January 2026*
