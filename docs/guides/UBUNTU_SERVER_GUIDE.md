# Ubuntu LLM Server Maintenance Guide

This guide covers the setup, maintenance, and troubleshooting of the Ubuntu LLM server at 10.101.20.21.

## Overview

| Property | Value |
|----------|-------|
| **Hostname/IP** | 10.101.20.21 |
| **Username** | chad |
| **GPUs** | RTX 3050 (8GB) + GTX 1660 (6GB) |
| **Purpose** | LLM inference servers, Docker containers |
| **SSH Key** | WSL: `~/.ssh/id_ed25519` |

## Connecting to the Server

```bash
# Run single command
ssh chad@10.101.20.21 "command here"

# Interactive session
ssh chad@10.101.20.21
```

## Common Maintenance Commands

### System Status

```bash
# Check GPU status
ssh chad@10.101.20.21 "nvidia-smi"

# Check memory usage
ssh chad@10.101.20.21 "free -h"

# Check disk space
ssh chad@10.101.20.21 "df -h"

# Check running services
ssh chad@10.101.20.21 "systemctl status llama*"

# Check Docker containers
ssh chad@10.101.20.21 "docker ps"
```

### Service Management

```bash
# View LLM service status
ssh chad@10.101.20.21 "systemctl status llama-sql"
ssh chad@10.101.20.21 "systemctl status llama-general"

# Restart a service
ssh chad@10.101.20.21 "sudo systemctl restart llama-sql"

# View service logs
ssh chad@10.101.20.21 "journalctl -u llama-sql -n 50"

# Follow logs in real-time
ssh chad@10.101.20.21 "journalctl -u llama-sql -f"
```

## Disk Layout

| Device | Size | Mount Point | Purpose |
|--------|------|-------------|---------|
| sda | 1.8T | (unmounted) | WD Red NAS drive (NTFS) |
| sdb1 | 100G | /home | User home directories |
| sdb2 | 500G | /data/models | LLM model storage |
| sdb3 | 500G | /data/projects | Project files (website) |
| sdb4 | 300G | /data/backups | Backup storage |
| sdc | 224G | / (SSD) | System drive |

## Removable Media Handling

### Detecting Connected Drives

```bash
# List all block devices with details
ssh chad@10.101.20.21 "lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL"

# List USB devices specifically
ssh chad@10.101.20.21 "lsusb"

# Watch for new device connections in real-time
ssh chad@10.101.20.21 "dmesg -w"

# Check recently connected devices
ssh chad@10.101.20.21 "dmesg | tail -30"

# List /dev/sd* devices (USB drives appear as sdb, sdc, sdd, etc.)
ssh chad@10.101.20.21 "ls -la /dev/sd*"
```

### Mounting a USB Drive

```bash
# Create mount point (if needed)
ssh chad@10.101.20.21 "sudo mkdir -p /mnt/usb"

# Mount the drive (replace sdX1 with actual partition)
ssh chad@10.101.20.21 "sudo mount /dev/sdX1 /mnt/usb"

# Mount with specific filesystem type
ssh chad@10.101.20.21 "sudo mount -t ntfs /dev/sdX1 /mnt/usb"   # NTFS
ssh chad@10.101.20.21 "sudo mount -t vfat /dev/sdX1 /mnt/usb"   # FAT32
ssh chad@10.101.20.21 "sudo mount -t exfat /dev/sdX1 /mnt/usb"  # exFAT

# Verify mount and list contents
ssh chad@10.101.20.21 "ls -la /mnt/usb"
```

### Safely Unmounting

```bash
# Sync any pending writes
ssh chad@10.101.20.21 "sync"

# Unmount the drive
ssh chad@10.101.20.21 "sudo umount /mnt/usb"

# Force unmount if busy (use with caution)
ssh chad@10.101.20.21 "sudo umount -l /mnt/usb"
```

### Copying Data To/From USB

```bash
# Copy files TO USB drive
ssh chad@10.101.20.21 "sudo cp /data/backups/myfile.tar.gz /mnt/usb/"

# Copy directory TO USB drive
ssh chad@10.101.20.21 "sudo cp -r /data/projects/backup /mnt/usb/"

# Copy files FROM USB drive
ssh chad@10.101.20.21 "sudo cp /mnt/usb/data.tar.gz /data/backups/"

# Sync large directories (shows progress)
ssh chad@10.101.20.21 "sudo rsync -avh --progress /data/models/ /mnt/usb/models/"
```

### Checking Drive Health

```bash
# Get drive information
ssh chad@10.101.20.21 "sudo hdparm -I /dev/sdX"

# Check SMART status (if supported)
ssh chad@10.101.20.21 "sudo smartctl -a /dev/sdX"

# Check filesystem for errors (unmount first)
ssh chad@10.101.20.21 "sudo fsck /dev/sdX1"
```

### Formatting a Drive

```bash
# WARNING: This erases all data on the drive!

# Format as ext4 (Linux native)
ssh chad@10.101.20.21 "sudo mkfs.ext4 /dev/sdX1"

# Format as FAT32 (universal compatibility, <4GB file limit)
ssh chad@10.101.20.21 "sudo mkfs.vfat -F 32 /dev/sdX1"

# Format as exFAT (large files, Windows/Mac/Linux compatible)
ssh chad@10.101.20.21 "sudo mkfs.exfat /dev/sdX1"

# Format as NTFS (Windows compatible)
ssh chad@10.101.20.21 "sudo mkfs.ntfs /dev/sdX1"
```

### Common USB Drive Issues

| Problem | Solution |
|---------|----------|
| Drive not detected | Check `dmesg` for errors, try different USB port |
| "Read-only filesystem" | Drive may have errors; run `fsck` on it |
| "Device is busy" | Close files/processes using the drive, then unmount |
| NTFS won't mount | Install `ntfs-3g`: `sudo apt install ntfs-3g` |
| exFAT won't mount | Install `exfat-fuse`: `sudo apt install exfat-fuse exfat-utils` |
| Slow transfer speeds | Use USB 3.0 port, check with `lsusb -t` for speed |

### Monitoring USB Events

```bash
# Watch for plug/unplug events
ssh chad@10.101.20.21 "udevadm monitor --subsystem-match=block"

# Check USB device tree with speed info
ssh chad@10.101.20.21 "lsusb -t"
```

## File Synchronization

The website runs from this server at `/data/projects/llm_website/`.

### Copying Files to Server

```bash
# Copy single file
scp "C:/projects/LLM_Website/public/js/myfile.js" chad@10.101.20.21:/data/projects/llm_website/public/js/

# Copy directory
scp -r "C:/projects/LLM_Website/public/admin/NewFeature" chad@10.101.20.21:/data/projects/llm_website/public/admin/
```

### Path Mapping

| Local Path | Server Path |
|------------|-------------|
| `C:\projects\LLM_Website\` | `/data/projects/llm_website/` |
| `C:\projects\LLM_Website\public\` | `/data/projects/llm_website/public/` |
| `C:\projects\LLM_Website\public\js\` | `/data/projects/llm_website/public/js/` |

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
ssh chad@10.101.20.21 "nvidia-smi"

# Reload NVIDIA modules
ssh chad@10.101.20.21 "sudo modprobe nvidia"

# Check for driver issues
ssh chad@10.101.20.21 "dmesg | grep -i nvidia"
```

### Service Won't Start

```bash
# Check service status and error
ssh chad@10.101.20.21 "systemctl status llama-sql"

# Check full logs
ssh chad@10.101.20.21 "journalctl -u llama-sql --no-pager"

# Check if port is in use
ssh chad@10.101.20.21 "sudo netstat -tlnp | grep 8080"
```

### Disk Space Issues

```bash
# Find large files
ssh chad@10.101.20.21 "sudo du -h /data --max-depth=2 | sort -hr | head -20"

# Clean Docker resources
ssh chad@10.101.20.21 "docker system prune -a"

# Clean apt cache
ssh chad@10.101.20.21 "sudo apt clean"
```

### Network Connectivity

```bash
# Check network interfaces
ssh chad@10.101.20.21 "ip addr"

# Test connectivity to other machines
ssh chad@10.101.20.21 "ping -c 3 10.101.20.29"  # EWRSPT-AI

# Check listening ports
ssh chad@10.101.20.21 "sudo netstat -tlnp"
```

## Backup Procedures

### Backup Models to USB

```bash
# Mount USB drive
ssh chad@10.101.20.21 "sudo mount /dev/sdX1 /mnt/usb"

# Sync models directory
ssh chad@10.101.20.21 "sudo rsync -avh --progress /data/models/ /mnt/usb/models-backup/"

# Unmount when done
ssh chad@10.101.20.21 "sync && sudo umount /mnt/usb"
```

### Backup Projects

```bash
# Create tarball
ssh chad@10.101.20.21 "sudo tar -czvf /data/backups/llm_website_$(date +%Y%m%d).tar.gz /data/projects/llm_website"

# Copy to USB
ssh chad@10.101.20.21 "sudo cp /data/backups/llm_website_*.tar.gz /mnt/usb/"
```

### MongoDB Automated Backup (Daily at 2 AM)

An automated backup job runs daily at 2 AM to back up MongoDB data to the USB drive.

**Configuration:**

| Setting | Value |
|---------|-------|
| **Schedule** | Daily at 2:00 AM |
| **Source** | MongoDB at 10.101.20.29:27018 |
| **Destination** | `/mnt/usb/mongodb_backups/` |
| **Retention** | 7 days |
| **Compression** | gzip |
| **Script** | `/data/projects/llm_website/scripts/mongodb_backup_usb.sh` |
| **Log File** | `/var/log/mongodb_backup.log` |

**Databases Backed Up:**
- `admin` - MongoDB system data
- `rag_server` - RAG server data (code analysis, SQL rules, documents)
- `llm_website` - LLM traces and website data

**Manual Backup:**
```bash
# Run backup manually
ssh chad@10.101.20.21 "sudo /data/projects/llm_website/scripts/mongodb_backup_usb.sh"

# Check backup log
ssh chad@10.101.20.21 "tail -50 /var/log/mongodb_backup.log"

# List existing backups
ssh chad@10.101.20.21 "ls -la /mnt/usb/mongodb_backups/"
```

**Restore from Backup:**
```bash
# Restore all databases
ssh chad@10.101.20.21 "mongorestore --host=10.101.20.29 --port=27018 --gzip /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/"

# Restore specific database
ssh chad@10.101.20.21 "mongorestore --host=10.101.20.29 --port=27018 --gzip --db=rag_server /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/"

# Restore specific collection
ssh chad@10.101.20.21 "mongorestore --host=10.101.20.29 --port=27018 --gzip --db=rag_server --collection=sql_rules /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/sql_rules.bson.gz"
```

**Check Cron Job:**
```bash
# View scheduled backup job
ssh chad@10.101.20.21 "sudo crontab -l | grep mongodb"

# Edit cron schedule
ssh chad@10.101.20.21 "sudo crontab -e"
```

**Requirements:**
- USB drive must be connected and mounted at `/mnt/usb`
- Requires at least 1GB free space on USB
- MongoDB tools installed (`mongodump`, `mongorestore`)

**Troubleshooting:**

| Problem | Solution |
|---------|----------|
| Backup fails - USB not found | Connect USB drive, check `/dev/sdd1` exists |
| Backup fails - no space | Delete old backups or use larger drive |
| Cron not running | Check `sudo crontab -l`, verify cron service running |
| Cannot connect to MongoDB | Verify MongoDB running on EWRSPT-AI, check network |

## Active Directory Integration

The server is joined to the **EWRINC.COM** domain. For full documentation on AD authentication and Windows share mounting, see:

**[Active Directory Integration Guide](ACTIVE_DIRECTORY_INTEGRATION.md)**

### Quick Reference

```bash
# Check domain status
ssh chad@10.101.20.21 "realm list"

# Test AD user resolution
ssh chad@10.101.20.21 "id chad.walker@ewrinc"

# Get Kerberos ticket
ssh chad@10.101.20.21 "kinit chad.walker@EWRINC.COM"

# Mount Windows share
ssh chad@10.101.20.21 "sudo mount -t cifs //ewrtnfile1/TNShare /mnt/tnshare -o sec=krb5,cruid=\$(id -u),vers=3.0"
```
