#!/bin/bash
#
# MongoDB Backup Script - Backs up MongoDB to USB drive
# Runs via cron at 2 AM daily
#
# Usage: ./mongodb_backup_usb.sh
#

set -e

# Configuration
MONGO_HOST="10.101.20.21"
MONGO_PORT="27017"
USB_MOUNT="/mnt/usb"
USB_DEVICE="/dev/sdd1"
BACKUP_DIR="${USB_MOUNT}/mongodb_backups"
LOG_FILE="/var/log/mongodb_backup.log"
RETENTION_DAYS=7

# Timestamp for this backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if running as root (needed for mount)
if [ "$EUID" -ne 0 ]; then
    error_exit "This script must be run as root (use sudo)"
fi

log "=========================================="
log "Starting MongoDB backup to USB"
log "=========================================="

# Check if USB device exists
if [ ! -b "$USB_DEVICE" ]; then
    error_exit "USB device $USB_DEVICE not found. Is the drive connected?"
fi

# Mount USB drive if not already mounted
if ! mountpoint -q "$USB_MOUNT"; then
    log "Mounting USB drive..."
    mkdir -p "$USB_MOUNT"
    mount "$USB_DEVICE" "$USB_MOUNT" || error_exit "Failed to mount USB drive"
    log "USB drive mounted at $USB_MOUNT"
else
    log "USB drive already mounted at $USB_MOUNT"
fi

# Check available space on USB (require at least 1GB)
AVAILABLE_KB=$(df "$USB_MOUNT" | awk 'NR==2 {print $4}')
AVAILABLE_MB=$((AVAILABLE_KB / 1024))
log "Available space on USB: ${AVAILABLE_MB} MB"

if [ "$AVAILABLE_MB" -lt 1024 ]; then
    error_exit "Insufficient space on USB drive (${AVAILABLE_MB} MB available, need at least 1024 MB)"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
log "Backup directory: $BACKUP_PATH"

# Run mongodump
log "Starting mongodump from ${MONGO_HOST}:${MONGO_PORT}..."
mongodump \
    --host="$MONGO_HOST" \
    --port="$MONGO_PORT" \
    --out="$BACKUP_PATH" \
    --gzip \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "Mongodump completed successfully"
else
    error_exit "Mongodump failed"
fi

# Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
log "Backup size: $BACKUP_SIZE"

# Create a latest symlink (skip on FAT32/VFAT which doesn't support symlinks)
if ln -sfn "$BACKUP_PATH" "${BACKUP_DIR}/latest" 2>/dev/null; then
    log "Updated 'latest' symlink"
else
    # Write latest path to a file instead
    echo "$BACKUP_PATH" > "${BACKUP_DIR}/LATEST.txt"
    log "Created LATEST.txt (symlinks not supported on this filesystem)"
fi

# Clean up old backups (keep last N days)
log "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -maxdepth 1 -type d -name "backup_*" -mtime +${RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null
REMAINING=$(ls -1d "${BACKUP_DIR}"/backup_* 2>/dev/null | wc -l)
log "Remaining backups: $REMAINING"

# Sync filesystem
sync
log "Filesystem synced"

# Summary
log "=========================================="
log "Backup completed successfully!"
log "Location: $BACKUP_PATH"
log "Size: $BACKUP_SIZE"
log "=========================================="

exit 0
